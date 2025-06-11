import boto3
import json
import pprint

client = boto3.client(service_name='bedrock-runtime', region_name="us-east-1")

titan_model_id = 'amazon.titan-text-express-v1'

titan_config = json.dumps({
            "inputText": "Crea un cuento sobre un pez",
            "textGenerationConfig": {
                "maxTokenCount": 100,
                "stopSequences": [],
                "temperature": 0,
                "topP": 1
            }
        })

response = client.invoke_model(
    body=titan_config,
    modelId=titan_model_id,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get('body').read())

pp = pprint.PrettyPrinter(depth=4)
# pp.pprint(response_body.get('results')) # titan config
pp.pprint(response_body["results"][0]["outputText"]) # llama config