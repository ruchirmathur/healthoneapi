import os
import json
import logging
import requests
from flask import Flask, request, jsonify
from flask_cors import cross_origin
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

API_URL = "http://localhost:8000"
AZURE_OPEN_AI_ENDPOINT = os.getenv("AZURE_OPEN_AI_ENDPOINT")
AZURE_OPEN_AI_KEY = os.getenv("AZURE_OPEN_AI_KEY")
API_VERSION = os.getenv("API_VERSION")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")

COSMOS_DB_URL_APPLICATION = os.getenv("COSMOS_DB_URL_APPLICATION")
COSMOS_DB_KEY_APPLICATION = os.getenv("COSMOS_DB_KEY_APPLICATION")
DATABASE_NAME_APPLICATION = os.getenv("DATABASE_NAME_APPLICATION")
CONTAINER_NAME_APPLICATION = os.getenv("CONTAINER_NAME_APPLICATION")

COSMOS_DB_URL_FEEDBACK = os.getenv("COSMOS_DB_URL_FEEDBACK")
COSMOS_DB_KEY_FEEDBACK = os.getenv("COSMOS_DB_KEY_FEEDBACK")
DATABASE_NAME_FEEDBACK = os.getenv("DATABASE_NAME_FEEDBACK")
CONTAINER_NAME_FEEDBACK = os.getenv("CONTAINER_NAME_FEEDBACK")

COSMOS_DB_URL_MHR = os.getenv("COSMOS_DB_URL_MHR")
COSMOS_DB_KEY_MHR = os.getenv("COSMOS_DB_KEY_MHR")
DATABASE_NAME_MHR = os.getenv("DATABASE_NAME_MHR")
CONTAINER_NAME_MHR = os.getenv("CONTAINER_NAME_MHR")

COSMOS_DB_URL_HOSPITAL = os.getenv("COSMOS_DB_URL_HOSPITAL")
COSMOS_DB_KEY_HOSPITAL = os.getenv("COSMOS_DB_KEY_HOSPITAL")
DATABASE_NAME_HOSPITAL = os.getenv("DATABASE_NAME_HOSPITAL")
CONTAINER_NAME_HOSPITAL = os.getenv("CONTAINER_NAME_HOSPITAL")

DOCUMENT_URL = os.getenv("DOCUMENT_URL")
DOCUMENT_KEY = os.getenv("DOCUMENT_KEY")

app = Flask(__name__)

openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPEN_AI_ENDPOINT,
    api_key=AZURE_OPEN_AI_KEY,
    api_version=API_VERSION
)

clientapplication = CosmosClient(COSMOS_DB_URL_APPLICATION, COSMOS_DB_KEY_APPLICATION)
databaseapplication = clientapplication.create_database_if_not_exists(id=DATABASE_NAME_APPLICATION)
containerapplication = databaseapplication.create_container_if_not_exists(
    id=CONTAINER_NAME_APPLICATION,
    partition_key=PartitionKey(path="/application_id"),
    offer_throughput=400
)

clientfeedback = CosmosClient(COSMOS_DB_URL_FEEDBACK, COSMOS_DB_KEY_FEEDBACK)
databasefeedback = clientfeedback.create_database_if_not_exists(id=DATABASE_NAME_FEEDBACK)
containerfeedback = databasefeedback.create_container_if_not_exists(
    id=CONTAINER_NAME_FEEDBACK,
    partition_key=PartitionKey(path="/portal_type"),
    offer_throughput=400
)

clientmhr = CosmosClient(COSMOS_DB_URL_MHR, COSMOS_DB_KEY_MHR)
databasemhr = clientmhr.create_database_if_not_exists(id=DATABASE_NAME_MHR)
containermhr = databasemhr.create_container_if_not_exists(
    id=CONTAINER_NAME_MHR,
    partition_key=PartitionKey(path="/member_id"),
    offer_throughput=400
)

clienthospital = CosmosClient(COSMOS_DB_URL_HOSPITAL, COSMOS_DB_KEY_HOSPITAL)
databasehospital = clienthospital.create_database_if_not_exists(id=DATABASE_NAME_HOSPITAL)
containerhospital = databasehospital.get_container_client(CONTAINER_NAME_HOSPITAL)

document_intelligence_client = DocumentIntelligenceClient(
    endpoint=DOCUMENT_URL, credential=AzureKeyCredential(DOCUMENT_KEY)
)

@app.route('/applications', methods=['POST'])
@cross_origin()
def get_applications():
    data = request.get_json() or {}
    query = "SELECT * FROM c"
    conditions = []
    parameters = []

    if "application_id" in data and data["application_id"] is not None:
        conditions.append("c.application_id = @application_id")
        parameters.append({"name": "@application_id", "value": data["application_id"]})
    if "status" in data and data["status"]:
        conditions.append("LOWER(c.status) = @status")
        parameters.append({"name": "@status", "value": str(data["status"]).lower()})
    if "name" in data and data["name"]:
        conditions.append("LOWER(c.name) = @name")
        parameters.append({"name": "@name", "value": str(data["name"]).lower()})
    if "type" in data and data["type"]:
        conditions.append("LOWER(c.type) = @type")
        parameters.append({"name": "@type", "value": str(data["type"]).lower()})
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    try:
        items = containerapplication.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        )
        results = list(items)
        return jsonify(results), 200
    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": str(e)}), 500

documents = [
    {"application_id": i, "doc_id": f"Doc-{i}"}
    for i in range(1, 101)
]

@app.route('/documents/summary', methods=['POST'])
@cross_origin()
def summarize_document():
    data = request.json
    app_id = data.get("application_id")
    doc = next((doc for doc in documents if doc["application_id"] == app_id), None)
    if doc:
        return jsonify(doc)
    return jsonify({"error": "Document not found"}), 404

@app.route('/feedbacks', methods=['POST'])
@cross_origin()
def get_feedbacks_by_portal():
    data = request.json or {}
    portal_type = data.get("portal_type")
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    if not portal_type or not start_date or not end_date:
        return jsonify({"error": "Missing required field(s): portal_type, start_date, end_date"}), 400

    query = (
        "SELECT * FROM c WHERE c.portal_type=@portal_type "
        "AND c.survey_date >= @start_date AND c.survey_date <= @end_date"
    )
    parameters = [
        {"name": "@portal_type", "value": portal_type},
        {"name": "@start_date", "value": start_date},
        {"name": "@end_date", "value": end_date},
    ]

    try:
        items = containerfeedback.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        )
        feedback_list = [item for item in items]
        return jsonify(feedback_list), 200
    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": str(e)}), 500

@app.route('/memberhealthrecord', methods=['POST'])
@cross_origin()
def member_health_records():
    data = request.json or {}
    member_id = data.get("member_id")
    if not member_id:
        return jsonify({"error": "Missing required field: member_id"}), 400

    query = "SELECT * FROM c WHERE c.member_id=@member_id"
    parameters = [{"name": "@member_id", "value": member_id}]
    try:
        items = containermhr.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        )
        mhr_list = [item for item in items]
        return jsonify(mhr_list), 200
    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": str(e)}), 500

def build_search_query(description, city=None, state=None):
    base_query = """
    SELECT  TOP 5 *
    FROM c
    WHERE EXISTS(
        SELECT VALUE sci
        FROM sci IN c.standard_charge_information
        WHERE CONTAINS(sci.description, @description, true)
    )
    """
    parameters = [{"name": "@description", "value": description}]
    location_filters = []
    if city:
        location_filters.append("CONTAINS(c.hospital_address[1], @city, true)")
        parameters.append({"name": "@city", "value": city})
    if state:
        location_filters.append("CONTAINS(c.hospital_address[2], @state, true)")
        parameters.append({"name": "@state", "value": state})
    if location_filters:
        base_query += " AND " + " AND ".join(location_filters)
    return base_query, parameters

@app.route('/hospital', methods=['POST'])
@cross_origin()
def search_hospital():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        description = data.get('description')
        city = data.get('city')
        state = data.get('state')

        if not description:
            return jsonify({"error": "Description is required"}), 400

        query, params = build_search_query(description, city, state)

        items = list(containerhospital.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True
        ))

        return jsonify(items), 200
    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def analyze_marriage_certificate(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdf_content = response.content

        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-marriageCertificate.us",
            body=pdf_content,
            content_type="application/octet-stream"
        )
        result = poller.result()
        extracted_data = {}
        for document in result.documents:
            fields = document.fields
            extracted_data = {
                "issue_date": get_field_value(fields.get("IssueDate")),
                "marriage_date": get_field_value(fields.get("MarriageDate")),
                "marriage_place": get_field_value(fields.get("MarriagePlace")),
                "spouse1": process_spouse_data(fields, "Spouse1"),
                "spouse2": process_spouse_data(fields, "Spouse2")
            }
            break
        return {
            "document_url": url,
            "details": extracted_data
        }
    except Exception as e:
        return {"error": str(e)}

def get_field_value(field):
    if not field:
        return None
    return {"value": field.content}

def process_spouse_data(fields, prefix):
    return {
        "first_name": get_field_value(fields.get(f"{prefix}FirstName")),
        "middle_name": get_field_value(fields.get(f"{prefix}MiddleName")),
        "last_name": get_field_value(fields.get(f"{prefix}LastName")),
        "age": get_field_value(fields.get(f"{prefix}Age")),
        "birth_place": get_field_value(fields.get(f"{prefix}BirthPlace")),
        "address": get_field_value(fields.get(f"{prefix}Address"))
    }

@app.route('/api/document-review', methods=['POST'])
@cross_origin()
def document_review():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "Missing PDF URL in request"}), 400

    result = analyze_marriage_certificate(data['url'])
    if 'error' in result:
        return jsonify(result), 500
    return jsonify(result)

# --- Azure Functions for LLM Tooling ---

def search_app_by_id(application_id=None, status=None, name=None, type_=None):
    logger.info("search_app_by_id called with application_id=%s, status=%s, name=%s, type_=%s", application_id, status, name, type_)
    try:
        payload = {}
        if application_id not in [None, "", []]:
            logger.debug("Attempting to cast application_id '%s' to int", application_id)
            try:
                payload["application_id"] = int(application_id)
                logger.debug("application_id successfully cast to int: %d", payload["application_id"])
            except (ValueError, TypeError) as e:
                logger.error("Invalid application_id '%s': %s", application_id, str(e))
                return {"error": "application_id must be an integer"}
        else:
            logger.debug("application_id not provided or empty; skipping.")

        if status:
            payload["status"] = status
            logger.debug("Added status to payload: %s", status)
        if name:
            payload["name"] = name
            logger.debug("Added name to payload: %s", name)
        if type_:
            payload["type"] = type_
            logger.debug("Added type to payload: %s", type_)

        logger.info("Final payload for /applications: %s", payload)
        url = f"{API_URL}/applications"
        logger.info("Sending POST request to %s", url)
        response = requests.post(url, json=payload)

        logger.debug("Received HTTP status code: %s", response.status_code)
        response.raise_for_status()
        logger.info("POST request to %s successful", url)

        try:
            data = response.json()
            logger.debug("Response JSON parsed successfully: %s", data)
        except Exception as e:
            logger.error("Failed to parse JSON response: %s", str(e))
            return {"error": "Failed to parse JSON response from /applications"}

        return data

    except requests.RequestException as e:
        logger.error("HTTP error during /applications request: %s", str(e), exc_info=True)
        return {"error": str(e)}
    except Exception as e:
        logger.error("Unexpected error in search_app_by_id: %s", str(e), exc_info=True)
        return {"error": str(e)}

def summarize_documents(application_id):
    response = requests.post(
        f"{API_URL}/documents/summary",
        json={"application_id": application_id}
    )
    response.raise_for_status()
    return response.json()

def get_feedbacks_by_portal(portal_type, start_date, end_date):
    if not all([portal_type, start_date, end_date]):
        return {"error": "Missing required fields: portal_type, start_date, end_date"}
    response = requests.post(
        f"{API_URL}/feedbacks",
        json={
            "portal_type": portal_type.capitalize(),
            "start_date": start_date,
            "end_date": end_date
        }
    )
    response.raise_for_status()
    return [{
        k: v for k, v in feedback.items()
        if k not in {"patterns", "sentiment", "potential_fixes"}
    } for feedback in response.json()]

def member_health_records_llm(member_id):
    response = requests.post(
        f"{API_URL}/memberhealthrecord",
        json={"member_id": member_id}
    )
    response.raise_for_status()
    return response.json()

def hospital(description=None, city=None, state=None):
    if not description:
        return {"error": "Procedure/service description is required"}
    payload = {"description": description}
    if city:
        payload["city"] = city
    if state:
        payload["state"] = state

    response = requests.post(f"{API_URL}/hospital", json=payload)
    response.raise_for_status()
    hospital_data = response.json()
    if not isinstance(hospital_data, list):
        hospital_data = [hospital_data]

    search_term = description.lower()
    filtered_results = []
    for hospital_entry in hospital_data:
        hospital_name = hospital_entry.get("hospital_name")
        hospital_address = hospital_entry.get("hospital_address")
        for proc in hospital_entry.get("standard_charge_information", []):
            if search_term in proc.get("description", "").lower():
                procedure_desc = proc.get("description")
                for charge in proc.get("standard_charges", []):
                    setting = charge.get("setting", hospital_entry.get("setting"))
                    for payer in charge.get("payers_information", []):
                        result = {
                            "hospital_name": hospital_name,
                            "hospital_address": hospital_address,
                            "setting": setting,
                            "standard_charge": {
                                "procedure": procedure_desc,
                                "setting": setting,
                                "payer_name": payer.get("payer_name"),
                                "plan_name": payer.get("plan_name"),
                                "price": payer.get("standard_charge_dollar"),
                                "methodology": payer.get("methodology")
                            }
                        }
                        filtered_results.append(result)
    return filtered_results if filtered_results else {"info": "No matching procedures found"}

azure_functions = [
    {
        "type": "function",
        "function": {
            "name": "search_app_by_id",
            "description": "Retrieve health insurance applications using various filters. Returns complete application details including status, documents, premiums, and qualifying life events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "application_id": {"type": "integer", "nullable": True},
                    "status": {"type": "string", "nullable": True},
                    "name": {"type": "string", "nullable": True},
                    "type": {"type": "string", "nullable": True},
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_documents",
            "description": "Retrieve summarized information for all documents associated with a specific application. Includes document types, summaries, and upload dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "application_id": {"type": "integer"}
                },
                "required": ["application_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_feedbacks_by_portal",
            "description": "Retrieve user feedback data filtered by portal type and date range. Includes feedback text, ratings, user IDs, statuses, and survey dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "portal_type": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"}
                },
                "required": ["portal_type", "start_date", "end_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "member_health_records_llm",
            "description": "Retrieve complete medical history for a member including diagnoses, treatments, prescriptions, and provider information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "member_id": {"type": "string"}
                },
                "required": ["member_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hospital",
            "description": "Retrieve detailed hospital pricing information for medical procedures. Includes payer-specific rates, discounts, payment settings, and effective dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string"}
                },
                "required": ["description"]
            }
        }
    }
]

def azure_function_dispatcher(function_name, function_args):
    if function_name == "search_app_by_id":
        return search_app_by_id(
            function_args.get("application_id"),
            function_args.get("status"),
            function_args.get("name"),
            function_args.get("type")
        )
    elif function_name == "summarize_documents":
        return summarize_documents(function_args["application_id"])
    elif function_name == "get_feedbacks_by_portal":
        return get_feedbacks_by_portal(
            function_args.get("portal_type"),
            function_args.get("start_date"),
            function_args.get("end_date")
        )
    elif function_name == "member_health_records_llm":
        return member_health_records_llm(function_args["member_id"])
    elif function_name == "hospital":
        return hospital(
            description=function_args.get("description"),
            city=function_args.get("city"),
            state=function_args.get("state")
        )
    return {"error": "Unknown function"}

def truncate_messages_to_fit(messages, max_prompt_tokens=6000):
    while sum(len(m.get("content", "")) for m in messages) > max_prompt_tokens and len(messages) > 1:
        messages.pop(0)
    return messages

combined_system_prompt = (
    "You are an expert assistant for applications, documents, feedback, health records, and hospitals. "
    "For every user query, always use the appropriate function(s) provided to retrieve information. "
    "After receiving data from a function, always summarize the results in clear, concise, and user-friendly language. "
    "For user feedback queries, in addition to the summary, analyze each comment, classify as positive, negative, or neutral, "
    "and provide the percentage of each sentiment. State the overall sentiment, summarize key issues, and give actionable recommendations. "
    "Return your summary as a string in 'final_response', and also output a JSON object called 'analysis' with these keys: "
    "'sentiment_percentages' (positive, negative, neutral), 'overall_sentiment', 'key_issues' (list), and 'recommendations' (list). "
    "For non-feedback queries, 'analysis' can be an empty object. "
    "If the user's request is ambiguous or missing details, ask clarifying questions before proceeding. "

    "For HOSPITAL PRICING queries: "
    "- Compare prices across providers. "
    "- Highlight best value options. "
    "- Include payer names, settings, and effective dates. "
    "- Output JSON with: array of price_comparison (hospital, plan, price, and setting), array of best_prices (address, hospital, plan, price, and setting), array of cost_saving_tips. "

    "For queries about claims, benefits, appointments, medications,dependents,allergies or immunization data: "
    "- Only include the relevant information in the JSON output. Always include address, birth_date,  coverage_effective_date, coverage_end_date, email, employer, gender, group_number,member_id, member_name, pcp_id, pcp_name, phone, plan_covers_procedure, plan_product_name, plan_type, subscriber, dependents in analysis field"
    "- Do not include unrelated data or sections. "
    "- Structure the JSON so that only the requested data type (claim_summary, allergies, benefits_summary, dependents, appointments, medications, or immunizations) is present in the analysis field "
    "- Member may ask for hospital pricing information, then call the hospital function and provide details"
    "- The 'analysis' field can be empty or contain a summary if appropriate. "
    "- The 'final_response' must summarize the requested information clearly. "

    "Never return raw JSON or lists of records without summarizing and explaining the key points, trends, or findings. "
    "Your goal is to make complex information easily understandable for the user. "
    "Always return a JSON object with both 'final_response' and 'analysis' as top-level keys, and do not use markdown or code blocks."
)

@app.route("/api/ask", methods=["POST"])
@cross_origin()
def ask():
    try:
        user_input = request.json["user_input"]
        messages = [
            {"role": "system", "content": combined_system_prompt},
            {"role": "user", "content": user_input}
        ]

        response = openai_client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=truncate_messages_to_fit(messages),
            tools=azure_functions,
            tool_choice="auto",
            response_format={"type": "json_object"},
            max_tokens=4000
        )
        msg = response.choices[0].message

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            new_messages = [
                {"role": "system", "content": combined_system_prompt},
                {"role": "user", "content": user_input},
                msg.to_dict()
            ]
            results = []

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                fn_result = azure_function_dispatcher(fn_name, fn_args)
                new_messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(fn_result)
                })
                results.append({"function": fn_name, "result": fn_result})

            final_response = openai_client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=new_messages,
                response_format={"type": "json_object"},
                max_tokens=4000
            )
            content = json.loads(final_response.choices[0].message.content)
        else:
            content = json.loads(msg.content)
            results = []

        return jsonify({
            "results": results,
            "final_response": content.get("final_response", ""),
            "analysis": content.get("analysis", {})
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "results": [],
            "analysis": {}
        }), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)
