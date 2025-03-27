from utility.prism import Prism

def generate_report(client_info, scenarios):
    prism_instance = Prism()

    prism_instance.add_client_info(client_info)

    for scenario in scenarios:
        prism_instance.add_scenario(scenario)
    
    results = prism_instance.run_all_scenarios()

    return {
        'scenarios_data': results
    }


    
