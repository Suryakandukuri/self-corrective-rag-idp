import requests
import json
import nltk
from nltk.corpus import stopwords

# creating a dictionary for metadata_store
metadata_store = {}
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# CKAN API Data Retrieval
def fetch_ckan_package_data():
    api_url = "https://ckandev.indiadataportal.com/api/3/action/package_search?q=organization%3Aidp-organization&rows=1000"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    }
    # Make the request
    response = requests.get(api_url, headers=headers)

    if response.status_code != 200:
        print(f"Error: Received HTTP {response.status_code}")
        print(f"Response text: {response.text}")
    else:
        try:
            json_data = response.json()
        except requests.exceptions.JSONDecodeError:
            print("Error: Response is not valid JSON")
            print(f"Raw response: {response.text}")

    if response.status_code == 200 and json_data.get("success"):
        packages = json_data["result"]["results"]
        documents = []
    for package in packages:
        for resource in package["resources"]:
            datastore_info_texts = []
            sku = resource.get("sku", "")
            resource_text = f"Resource Name: {resource.get('name', '')}, Format: {resource.get('format', '')}, Description: {resource.get('description', '')}, Data_Insights: {resource.get('data_insights', '')},methodology: {resource.get('methodology', '')}, Data_Usage: {resource.get('data_usage', '')},frequency: {resource.get('frequency', '')}, sku: {resource.get('sku', '')},data_last_updated: {resource.get('data_last_updated', '')}, data_retreival_date: {resource.get('data_retreival_date', '')}"
            api_url = f"https://ckandev.indiadataportal.com/api/3/action/datastore_info?id={resource['id']}"
            response = requests.get(api_url).json()
            # unwanted columns names
            unwanted_cols = ["id","year","index","state_name", "state_code","district_name","district_code","subdistrict_name", "subdistrict_code","block_name", "block_code", "gp_name","gp_code"]
            rows = response.get("result", {}).get("fields", [])
            # remove unwanted cols from rows (list of dicts) where row["id"] is not in unwanted cols
            if len(rows) > 0: 
                print(resource["id"])
                filtered_fields = [field["info"]["label"] if "info" in field else field["id"] for field in rows if field["id"] not in unwanted_cols]
            else:
                pass
            datastore_info_texts.append(" ".join(str(field) for field in filtered_fields))
            # Store full metadata separately
            metadata = {
                "package_id": package["id"],
                "title": package["title"],
                "url": package["url"],
                "package_name": package["name"],
                "sku": sku
            }
            metadata_store[sku] = metadata
        combined_text = f"{package['title']} {package['notes']} {package['name']} {package['source_name']} {package['sector']} {resource_text} {datastore_info_texts}"
        
        documents.append(
            {
                "text": preprocess_text(combined_text),
                "metadata": {
                    "sku": sku
                },
            }
        )
            # Save metadata store to disk
    with open("metadata_store.json", "w") as f:
        json.dump(metadata_store, f)
    return documents


# Fetch and combine metadata from resources
def fetch_resource_details(resources):
    resource_texts = []
    sku_list = []
    for resource in resources:
        sku_list.append(resource.get("sku", ""))
        resource_text = f"Resource Name: {resource.get('name', '')}, Format: {resource.get('format', '')}, Description: {resource.get('description', '')}, Data_Insights: {resource.get('data_insights', '')},methodology: {resource.get('methodology', '')}, Data_Usage: {resource.get('data_usage', '')},frequency: {resource.get('frequency', '')}, sku: {resource.get('sku', '')},data_last_updated: {resource.get('data_last_updated', '')}, data_retreival_date: {resource.get('data_retreival_date', '')}"
        resource_texts.append(resource_text)
    return " ".join(resource_texts), sku_list


# Datastore SQL API Retrieval (Example for query fetching columns)
def fetch_datastore_info(resources):
    datastore_info_texts = []
    for resource in resources:
        api_url = f"https://ckandev.indiadataportal.com/api/3/action/datastore_info?id={resource['id']}"
        response = requests.get(api_url).json()
        rows = response.get("result", {}).get("records", [])
        datastore_info_texts.append(" ".join(str(row) for row in rows))
        return datastore_info_texts


# Custom preprocessing logic
def preprocess_text(text):
    # Convert to lowercase and strip extra spaces
    text = text.lower().strip()

    # Tokenize the text and remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]

    # Join the tokens back into a string
    return " ".join(tokens)