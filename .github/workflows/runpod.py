import requests
import logging
import time
import sys
import os

logging.basicConfig(level=logging.INFO)

def get_endpoints():
    return f"""
    query {{
      myself {{
        endpoints {{
          id
          name
          templateId
          gpuIds
          workersMax
          workersMin
          networkVolumeId
        }}
      }}
    }}
    """

def create_or_update_template(template_id, name, container_disk_in_gb, image_name, env_vars, readme):
    input_fields = []

    if template_id:
        input_fields.append(f'id: "{template_id}"') 

    input_fields.append(f'name: "{name}"')
    input_fields.append(f'containerDiskInGb: {container_disk_in_gb}') 
    input_fields.append(f'imageName: "{image_name}"')
    input_fields.append(f'readme: "{readme}"')

    env = []
    for key, value in env_vars.items():
        env.append(f'{{ key: "{key}", value: "{value}" }}')
    input_fields.append(f'env: [{", ".join(env)}]')

    input_fields.append('volumeInGb: 0')
    input_fields.append('dockerArgs: ""')
    input_fields.append('isServerless: true')

    input_fields_string = ", ".join(input_fields)

    return f"""
    mutation {{
      saveTemplate(input: {{
        {input_fields_string}
      }}) {{
        id
        imageName
        name
      }}
    }}
    """

def create_or_update_endpoint(endpoint_id, endpoint_name, template_id, gpu_ids, workers_max, workers_min, network_volume_id):
    input_fields = []

    if endpoint_id:
        input_fields.append(f'id: "{endpoint_id}"') 

    input_fields.append(f'name: "{endpoint_name}"')
    input_fields.append(f'templateId: "{template_id}"')
    input_fields.append(f'gpuIds: "{gpu_ids}"') 
    input_fields.append(f'workersMin: {workers_min}')
    input_fields.append(f'workersMax: {workers_max}')

    if network_volume_id is not None:
        input_fields.append(f'networkVolumeId: "{network_volume_id}"')
    else:
        input_fields.append('networkVolumeId: ""')

    input_fields_string = ", ".join(input_fields)

    return f"""
    mutation {{
      saveEndpoint(input: {{
        {input_fields_string}
      }}) {{
        id
        name
      }}
    }}
    """

def update_endpoint_min_workers(endpoint_id, worker_count):
    return f"""
    mutation {{
      updateEndpointWorkersMin(input: {{
        endpointId: "{endpoint_id}",
        workerCount: {worker_count}
      }}) {{
        id
      }}
    }}
    """

def update_endpoint_max_workers(endpoint_id, worker_count):
    return f"""
    mutation {{
      updateEndpointWorkersMax(input: {{
        endpointId: "{endpoint_id}",
        workerCount: {worker_count}
      }}) {{
        id
      }}
    }}
    """

def make_graphql_request(query):
    api_key = os.environ.get('RUNPOD_TOKEN')
    runpod_url = f"https://api.runpod.io/graphql?api_key={api_key}"

    try:
        response = requests.post(url=runpod_url, json={"query": query})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"GraphQL request failed: {e}")
        return None

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "update"

    required_env_vars = {
        "promote": [
            "RUNPOD_TOKEN",
            "RUNPOD_SOURCE_ENDPOINT",
            "RUNPOD_TARGET_ENDPOINT"
        ],
        "update": [
            "RUNPOD_TOKEN",
            "RUNPOD_IMAGE",
            "RUNPOD_ENDPOINT",
        ]
    }

    required_vars = required_env_vars[action]

    missing_env_vars = [env_var for env_var in required_vars if not os.environ.get(env_var)]

    if missing_env_vars:
        logging.error(f"The following environment variables are not set: {', '.join(missing_env_vars)}")
        logging.error("The script cannot continue without these variables.")
        sys.exit(1)

    endpoints = make_graphql_request(get_endpoints())['data']['myself']['endpoints']

    match action:
        case "update":
            endpoint_name = os.environ.get("RUNPOD_ENDPOINT")
            image =  os.environ.get("RUNPOD_IMAGE")
            github_action_run = os.environ.get("GITHUB_RUN_NUMBER", "0")
            safe_deploy = os.environ.get("SAFE_DEPLOY", "true")
            env_vars = {key.replace("RUNPOD_VAR_", ""): value for key, value in os.environ.items() if key.startswith("RUNPOD_VAR_")}

            name = image.split('/')[-1]
            readme = f"Generated by Github Actions #{github_action_run}"

            endpoint_id = ""
            template_id = ""
            for ep in endpoints:
                if ep['name'] == f"{endpoint_name}-preview":
                    endpoint_id = ep['id']
                    template_id = ep['templateId']
                elif ep['name'] == endpoint_name:
                    endpoint = ep

            if safe_deploy.lower() == "true":
                logging.info("Using SAFE_DEPLOY: A preview endpoint will be created.")
                endpoint['name'] += "-preview"
                endpoint['templateId'] = template_id
                endpoint['id'] = endpoint_id

            if endpoint_id:
                logging.info(f"Setting the worker count to zero for endpoint {endpoint['id']} before the update.")
                make_graphql_request(update_endpoint_max_workers(endpoint['id'], 0))
                make_graphql_request(update_endpoint_min_workers(endpoint['id'], 0))
                time.sleep(5)

            endpoint['templateId'] = make_graphql_request(create_or_update_template(
                endpoint['templateId'],
                name,
                100,
                image,
                env_vars,
                readme)
            )['data']['saveTemplate']['id']

            make_graphql_request(create_or_update_endpoint(*endpoint.values()))
        case "promote":
            source_name = os.environ.get("RUNPOD_SOURCE_ENDPOINT")
            target_name = os.environ.get("RUNPOD_TARGET_ENDPOINT")
    
            source, target = [next(endpoint for endpoint in endpoints if endpoint['name'] == name) for name in [source_name, target_name]]
    
            source['name'], target['name'] = target['name'], source['name']
    
            make_graphql_request(create_or_update_endpoint(*source.values()))
            make_graphql_request(create_or_update_endpoint(*target.values()))

            logging.info(f"The preview endpoint with ID {source['id']} has been promoted to the main endpoint.")
            logging.info(f"The previously used endpoint has ID: {target['id']}")

            with open(os.environ['GITHUB_OUTPUT'], "a") as fh:
                print(f"endpoint={source['id']}", file=fh)
