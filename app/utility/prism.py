from typing import Dict, List, Tuple, Optional
import multiprocessing
import concurrent.futures
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import riskfolio as rf

class Prism:
    """
    Prism AI Wealth Management System
    
    This class encapsulates functionality for:
    - Calculating efficient frontiers
    - Running Monte Carlo simulations
    - Generating asset allocation visualizations
    - Running multiple investment scenarios
    """
    
    def __init__(self):
        # System settings
        self.number_of_simulation = 200  # Number of Monte Carlo simulations
        self.num_processers = max(int(multiprocessing.cpu_count() * 0.5), 1) # Set to use 50% CPU Power
        self.num_points = 100  # Number of efficient frontier points
        
        # Load CMA data
        self.cma = pd.read_csv('./app/Prism_CMA_Data.csv')
        self.asset_class_name = self.cma['Asset Class'].values.tolist()
        self.asset_class_return = self.cma['Long Term Annual Return'].values.tolist()
        self.standard_deviations = self.cma['Risk'].values.tolist()
        self.num_assets = len(self.asset_class_name)
        
        # Load correlation matrix and calculate covariance
        self.correlation_matrix = pd.read_csv('./app/correlation_matrix.csv').set_index('Asset Class').values
        sigma = np.diag(self.standard_deviations)
        self.covariance_matrix = sigma @ self.correlation_matrix @ sigma
        
        # Asset class and liquidity data
        self.asset_class_loading = self.cma[['Asset Class', 'Asset Group']].copy()
        self.liquidity_loading = self.cma[['Liquidity Score']].copy()
        
        # Define risk and liquidity mappings
        self.risk_map = {
            'Very-Conservative': 1.0,
            'Conservative': 2.5,
            'Conservative-Moderate': 4.0,
            'Moderate': 5.5,
            'Moderate-Aggressive': 7.0,
            'Aggressive': 8.5,
        }
        
        self.liquidity_map = {
            'Short-term': [0, 10.0],
            'Moderate': [0, 6.0],
            'Long-term': [0, 3.0],
        }
        
        # Client information
        self.client_info = None
        
        # Scenario definitions
        self.scenarios = {}

    def add_client_info(self, client_info: Dict) -> None:
        """
        Add client info
        
        Args:
            client_info: Dictionary containing client data
        """
        self.client_info = client_info

    def add_scenario(self, scenario_data: Dict) -> None:
        """
        Add a new scenario to the analysis
        
        Args:
            scenario_data: Dictionary containing scenario parameters
        """
        if 'scenario_id' not in scenario_data:
            raise ValueError("scenario_id is required in scenario_data")
        
        self.scenarios[scenario_data['scenario_id']] = scenario_data

    def build_asset_class_constraints(self, scenario_inputs: Dict) -> Optional[pd.DataFrame]:
        """
        Build asset class constraints dataframe from scenario inputs
        
        Args:
            scenario_inputs: Scenario input dictionary
            
        Returns:
            DataFrame with constraint definitions or None if no constraints
        """
        asset_class_constraints = scenario_inputs['constraints']['asset_class']
        asset_group_constraints = scenario_inputs['constraints']['asset_group']
        data = []
        for item in asset_class_constraints:
            asset_class = item['name']
            min_weight = item['min']
            max_weight = item['max']
            
            if min_weight != 0:
                data.append([False, 'Assets', '', asset_class, '>=', min_weight, '', '', '', ''])
            if max_weight != 1.0:
                data.append([False, 'Assets', '', asset_class, '<=', max_weight, '', '', '', ''])

        for item in asset_group_constraints:
            asset_group = item['name']
            min_weight = item['min']
            max_weight = item['max']
            
            if min_weight != 0:
                data.append([False, 'Classes', 'Asset Group', asset_group, '>=', min_weight, '', '', '', ''])
            if max_weight != 1.0:
                data.append([False, 'Classes', 'Asset Group', asset_group, '<=', max_weight, '', '', '', ''])
                
        if not data:
            return None
            
        return pd.DataFrame(
            data, 
            columns=['Disabled', 'Type', 'Set', 'Position', 'Sign', 'Weight', 
                     'Type Relative', 'Relative Set', 'Relative', 'Factor']
        )

    def build_liquidity_constraints(self, scenario_inputs: Dict) -> pd.DataFrame:
        """
        Build liquidity constraints dataframe from scenario inputs
        
        Args:
            scenario_inputs: Scenario input dictionary
            
        Returns:
            DataFrame with liquidity constraints
        """
        liquidity_level = scenario_inputs['liquidity_level']
        liquidity_level_min, liquidity_level_max = self.liquidity_map[liquidity_level]
        
        return pd.DataFrame({
            'Disabled': [False, False],
            'Factor': ['Liquidity Score', 'Liquidity Score'],
            'Sign': ['>=', '<='],
            'Value': [liquidity_level_min, liquidity_level_max],
            'Relative Factor': ['', '']
        })

    def calculate_efficient_frontier(self, scenario_input: Dict) -> Tuple[List, List, List]:
        """
        Calculate the efficient frontier for a given scenario using SciPy optimize
        
        Args:
            scenario_input: Scenario input dictionary
            
        Returns:
            Tuple of (returns, risks, weights) for the efficient frontier
        """
        asset_class_constraints = self.build_asset_class_constraints(scenario_input)
        liquidity_constraints = self.build_liquidity_constraints(scenario_input)

        A, B = rf.assets_constraints(asset_class_constraints, self.asset_class_loading)
        C, D = rf.factors_constraints(liquidity_constraints, self.liquidity_loading)

        returns = np.array(self.asset_class_return)
        risk = self.covariance_matrix
        n_assets = self.num_assets

        def portfolio_return(weights):
            return np.sum(returns * weights)

        def portfolio_risk(weights):
            return np.sqrt(np.dot(weights.T, np.dot(risk, weights)))

        def objective_max_return(weights):
            return -portfolio_return(weights)  # Negative because we're minimizing

        def objective_min_risk(weights):
            return portfolio_risk(weights)

        # Base constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]

        # Add asset class constraints
        if A is not None and B is not None:
            for i in range(len(B)):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, a=A[i], b=B[i]: b - np.dot(a, x)
                })

        # Add liquidity constraints
        if C is not None and D is not None:
            for i in range(len(D)):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, c=C[i], d=D[i]: d - np.dot(c, x)
                })

        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)  # Initial guess

        # Find maximum return portfolio
        result_max_return = minimize(
            objective_max_return,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        max_return_weights = result_max_return.x

        # Find minimum risk portfolio
        result_min_risk = minimize(
            objective_min_risk,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        min_risk_weights = result_min_risk.x

        # Create efficient frontier
        return_range = [portfolio_return(min_risk_weights), portfolio_return(max_return_weights)]
        efx, efy, efw = [], [], []
        
        for target_return in np.linspace(return_range[0], return_range[1], self.num_points):
            # Add return constraint
            current_constraints = constraints + [{
                'type': 'eq',
                'fun': lambda x, tr=target_return: portfolio_return(x) - tr
            }]
            
            # Optimize for minimum risk at target return
            result = minimize(
                objective_min_risk,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=current_constraints
            )
            
            if result.success:
                ef_weights = result.x
                ef_returns = portfolio_return(ef_weights)
                ef_risk = portfolio_risk(ef_weights)
                efx.append(ef_returns)
                efy.append(ef_risk)
                efw.append(ef_weights)
        
        return efx, efy, efw
    
    def run_monte_carlo(self, simulation_input):
        data_list = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processers) as executor:
            futures = [executor.submit(self.run_single_simulation, simulation_input) for _ in range(self.number_of_simulation)]
            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                data_list.append(data)
        df = pd.DataFrame(data_list, columns=range(self.client_info['Investment Start Year'],
                                                   self.client_info['Investment End Year'] + 2)).T
        return df

    def run_single_simulation(self, simulation_input: Dict) -> List:
        """
        Run a single Monte Carlo simulation
        
        Args:
            client_info: Dictionary with client info
            simulation_input: Dictionary with simulation parameters
            
        Returns:
            List of portfolio values over time
        """
        starting_asset_amount = self.client_info['Asset Amount']
        portfolio_return_overtime = simulation_input['portfolio_return_overtime']
        portfolio_risk_overtime = simulation_input['portfolio_risk_overtime']
        investment_starting_year = self.client_info['Investment Start Year']
        investment_ending_year = self.client_info['Investment End Year']
        investment_horizon = investment_ending_year - investment_starting_year + 1
        
        # Generate random returns based on expected return and risk
        portfolio_expected_return = [
            norm.ppf(np.random.rand(), loc=portfolio_return_overtime[i], scale=portfolio_risk_overtime[i]) 
            for i in range(investment_horizon)
        ]
        
        # Initialize model dataframe
        model_df = pd.DataFrame(
            [], 
            index=range(investment_starting_year, investment_ending_year + 1), 
            columns=['Starting Value']
        )
        model_df.loc[investment_starting_year, 'Starting Value'] = starting_asset_amount
        model_df['Expected Return'] = portfolio_expected_return
        
        # Add contributions
        for contribution in simulation_input['contributions']:
            model_df[contribution['id']] = 0.0
            data = [contribution['growth_rate']] * (contribution['ending_year'] - contribution['starting_year'] + 1)
            data[0] = 0
            data = pd.Series(data) + 1.0
            data = data.cumprod() * contribution['starting_amount']
            model_df.loc[contribution['starting_year']:contribution['ending_year'], contribution['id']] = data.values.tolist()
        
        # Add goals/spending
        for spending in simulation_input['goals']:
            model_df[spending['id']] = 0.0
            data = [spending['growth_rate']] * (spending['ending_year'] - spending['starting_year'] + 1)
            data[0] = 0
            data = pd.Series(data) + 1.0
            data = data.cumprod() * spending['starting_amount'] * -1.0
            model_df.loc[spending['starting_year']:spending['ending_year'], spending['id']] = data.values.tolist()
        
        # Calculate portfolio value over time
        model_df['Ending Value'] = 0.0
        model_df['Annual Net Income'] = 0.0
        contribution_columns = [contribution['id'] for contribution in simulation_input['contributions']]
        spending_columns = [spending['id'] for spending in simulation_input['goals']]
        
        for year in range(investment_starting_year, investment_ending_year + 1):
            starting_value = model_df.loc[year, 'Starting Value']
            expected_return = model_df.loc[year, 'Expected Return']
            net_income = model_df.loc[year, contribution_columns + spending_columns].sum()
            ending_value = starting_value * (1 + expected_return) + net_income
            model_df.loc[year, 'Ending Value'] = ending_value
            model_df.loc[year, 'Annual Net Income'] = net_income
            
            if year != investment_ending_year:
                model_df.loc[year + 1, 'Starting Value'] = ending_value
                
        return [starting_asset_amount] + model_df['Ending Value'].values.tolist()

    def run_scenario(self, scenario_id: str) -> Dict:
        """
        Run a complete scenario analysis and return data for frontend rendering
        
        Args:
            scenario_id: ID of the scenario to run
            
        Returns:
            Dictionary containing scenario metrics and graph data
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_id}' not found")
            
        simulation_input = self.scenarios[scenario_id]
        efx, efy, efw = self.calculate_efficient_frontier(simulation_input)

        # Select portfolio based on risk score
        risk_score = self.risk_map[simulation_input['risk_level']]
        idx = int((risk_score - 1.0) * ((self.num_points - 1) / 9))
        scenario_ptf_return = efx[idx]
        scenario_ptf_risk = efy[idx]
        scenario_ptf_weights = efw[idx]
        
        # Prepare pie chart data
        pie_graph = self.asset_class_loading.copy()
        pie_graph['weights'] = scenario_ptf_weights
        pie_data = pie_graph.groupby('Asset Group')['weights'].sum().to_dict()
        
        # Prepare bar chart data
        bar_graph = self.asset_class_loading.copy()
        bar_graph['weights'] = scenario_ptf_weights
        bar_data = bar_graph[['Asset Class', 'weights']].set_index('Asset Class')['weights'].to_dict()
        
        # Prepare efficient frontier data
        efficient_frontier_data = {
            'frontier_points': list(zip(efy, efx)),  # All points on frontier
            'selected_point': {
                'risk': scenario_ptf_risk,
                'return': scenario_ptf_return
            }
        }
        
        # Calculate dynamic asset allocation
        investment_horizon = list(range(self.client_info['Investment Start Year'], 
                                        self.client_info['Investment End Year'] + 1))
        risk_over_time = simulation_input['risk_over_time']
        
        # Determine risk score evolution over time
        if risk_over_time == 'Increase':
            risk_score_range = np.linspace(risk_score, 9.99, len(investment_horizon))
        elif risk_over_time == 'Moderate-Increase':
            risk_score_range = np.linspace(risk_score, min(risk_score + 3.0, 9.99), len(investment_horizon))
        elif risk_over_time == 'Stable':
            risk_score_range = np.linspace(risk_score, risk_score, len(investment_horizon))
        elif risk_over_time == 'Moderate-Decrease':
            risk_score_range = np.linspace(risk_score, max(risk_score - 3.0, 0.01), len(investment_horizon))
        elif risk_over_time == 'Decrease':
            risk_score_range = np.linspace(risk_score, 0.01, len(investment_horizon))
        else:
            raise ValueError("Risk Over Time Value Error")
        
        # Calculate portfolio return and risk over time
        ef_index_over_time = [int((rs - 1.0) * ((self.num_points - 1) / 9)) for rs in risk_score_range]
        simulation_input['portfolio_return_overtime'] = [efx[i] for i in ef_index_over_time]
        simulation_input['portfolio_risk_overtime'] = [efy[i] for i in ef_index_over_time]
        
        # Prepare dynamic allocation data
        dynamic_allocation_data = [efw[i] for i in ef_index_over_time]
        dynamic_allocation = pd.DataFrame(dynamic_allocation_data).T
        dynamic_allocation.index = self.asset_class_name
        dynamic_allocation.columns = investment_horizon
        
        dynamic_allocation_data = {
            'years': investment_horizon,
            'allocations': dynamic_allocation.to_dict('index')
        }
        
        # Run Monte Carlo simulations
        df = self.run_monte_carlo(simulation_input)
        
        # Prepare cash flow simulation data
        percentiles = df.quantile([0.1, 0.5, 0.75, 0.9], axis=1)
        cash_flow_data = {
            'years': df.index.tolist(),
            'percentiles': {
                '10th': percentiles.loc[0.1].tolist(),
                '50th': percentiles.loc[0.5].tolist(),
                '75th': percentiles.loc[0.75].tolist(),
                '90th': percentiles.loc[0.9].tolist()
            }
        }
        
        # Calculate metrics
        end_value = df.iloc[-1]
        metrics = {
            'expected_return': scenario_ptf_return,
            'portfolio_risk': scenario_ptf_risk,
            'shortfall_risk': (end_value < 0).sum() / len(end_value),
            'median_ending_value': end_value.median(),
            '10th_percentile_ending_value': end_value.quantile(0.10)
        }
        
        # Return all data
        return {
            'scenario_id': scenario_id,
            'scenario_name': simulation_input['scenario_name'],
            'metrics': self.format_metrics(metrics),
            'allocation_table_data': bar_data,
            'cash_flow_simulation_data': cash_flow_data,
            'dynamic_allocation_data': dynamic_allocation_data,
            'efficient_frontier_data': efficient_frontier_data,
            'pie_chart_data': pie_data
        }

    def format_metrics(self, metrics: Dict) -> Dict:
        """
        Format metrics for display
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Dictionary of formatted metrics
        """
        return {
            'expected_return': "{:.2%}".format(metrics['expected_return']),
            'portfolio_risk': "{:.2%}".format(metrics['portfolio_risk']),
            'shortfall_risk': "{:.2%}".format(metrics['shortfall_risk']),
            'median_ending_value': "${:,.2f}M".format(metrics['median_ending_value'] / 1e6),
            '10th_percentile_ending_value': "${:,.2f}M".format(metrics['10th_percentile_ending_value'] / 1e6)
        }

    def format_contributions(self, contributions: List[Dict]) -> str:
        """
        Format contributions for display
        
        Args:
            contributions: List of contribution dictionaries
            
        Returns:
            Formatted string
        """
        return ", ".join([
            f"{c['id']} (${c['starting_amount']:,.0f}, {c['starting_year']}-{c['ending_year']}, growth {c['growth_rate']:.2%})" 
            for c in contributions
        ])

    def format_goals(self, goals: List[Dict]) -> str:
        """
        Format goals for display
        
        Args:
            goals: List of goal dictionaries
            
        Returns:
            Formatted string
        """
        return ", ".join([
            f"{g['id']} (${g['starting_amount']:,.0f}, {g['starting_year']}-{g['ending_year']}, growth {g['growth_rate']:.2%})" 
            for g in goals
        ])

    def run_all_scenarios(self) -> Dict:
        """
        Run all scenarios and return formatted results
        
        Returns:
            Dictionary of scenario metrics
        """
        results = {}
        for scenario_id in self.scenarios:
            scenario_data = self.run_scenario(scenario_id)
            results[scenario_id] = scenario_data

        return results
