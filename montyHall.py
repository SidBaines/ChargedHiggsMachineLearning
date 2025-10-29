import random
import matplotlib.pyplot as plt

def monty_hall_simulation(num_trials=10000, num_doors=3, always_switch=None, debug=False):
    """
    Simulates the Monty Hall problem.
    
    Parameters:
    -----------
    num_trials : int
        Number of games to simulate
    num_doors : int
        Number of doors in the game (classic version uses 3)
    always_switch : bool or None
        If True, always switch doors
        If False, never switch doors
        If None, run both strategies and compare
    debug : bool
        If True, print detailed information about a few sample games
    
    Returns:
    --------
    Dictionary containing results of the simulation
    """
    # Initialize counters for statistics
    switch_wins = 0
    switch_losses = 0
    stay_wins = 0
    stay_losses = 0
    
    # Run the simulation for the specified number of trials
    for trial in range(num_trials):
        # Set up the game
        doors = list(range(num_doors))
        prize_door = random.choice(doors)
        initial_choice = random.choice(doors)
        
        # Host opens a door that's not the prize and not the initial choice
        available_doors = [door for door in doors if door != prize_door and door != initial_choice]
        
        # If the player initially picked the prize door, the host can open any of the other doors
        if not available_doors:
            available_doors = [door for door in doors if door != initial_choice]
        
        opened_door = random.choice(available_doors)
        
        # Remaining door that could be switched to
        remaining_doors = [door for door in doors if door != initial_choice and door != opened_door]
        door_to_switch_to = random.choice(remaining_doors)
        
        # Print debug information for a few sample games
        if debug and trial < 5:
            print(f"\nTrial {trial + 1}:")
            print(f"Prize is behind door {prize_door}")
            print(f"Player initially chooses door {initial_choice}")
            print(f"Host opens door {opened_door}")
            print(f"Player could switch to door {door_to_switch_to}")
        
        # Track results for switching strategy
        if door_to_switch_to == prize_door:
            switch_wins += 1
        else:
            switch_losses += 1
        
        # Track results for staying strategy
        if initial_choice == prize_door:
            stay_wins += 1
        else:
            stay_losses += 1
    
    # Calculate win rates
    switch_win_rate = switch_wins / num_trials * 100
    stay_win_rate = stay_wins / num_trials * 100
    
    # Prepare results
    results = {
        'num_trials': num_trials,
        'num_doors': num_doors,
        'switch_wins': switch_wins,
        'switch_losses': switch_losses,
        'switch_win_rate': switch_win_rate,
        'stay_wins': stay_wins,
        'stay_losses': stay_losses,
        'stay_win_rate': stay_win_rate
    }
    
    # If always_switch is specified, only return the relevant results
    if always_switch is True:
        return {'wins': switch_wins, 'losses': switch_losses, 'win_rate': switch_win_rate}
    elif always_switch is False:
        return {'wins': stay_wins, 'losses': stay_losses, 'win_rate': stay_win_rate}
    
    return results

def print_results(results):
    """
    Prints the results of the Monty Hall simulation in a readable format.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing simulation results
    """
    print("\n===== MONTY HALL PROBLEM SIMULATION RESULTS =====")
    print(f"Number of trials: {results['num_trials']}")
    print(f"Number of doors: {results['num_doors']}")
    
    print("\nSTAY STRATEGY:")
    print(f"Wins: {results['stay_wins']} ({results['stay_win_rate']:.2f}%)")
    print(f"Losses: {results['stay_losses']}")
    
    print("\nSWITCH STRATEGY:")
    print(f"Wins: {results['switch_wins']} ({results['switch_win_rate']:.2f}%)")
    print(f"Losses: {results['switch_losses']}")
    
    print("\nCONCLUSION:")
    if results['switch_win_rate'] > results['stay_win_rate']:
        difference = results['switch_win_rate'] - results['stay_win_rate']
        print(f"Switching doors is better by {difference:.2f} percentage points!")
        print(f"The theoretical advantage is {(results['num_doors']-1)/results['num_doors']*100:.2f}% for switching vs {1/results['num_doors']*100:.2f}% for staying.")
    elif results['stay_win_rate'] > results['switch_win_rate']:
        difference = results['stay_win_rate'] - results['switch_win_rate']
        print(f"Staying with the initial choice is better by {difference:.2f} percentage points!")
    else:
        print("Both strategies performed equally in this simulation.")

def plot_results(results, save_file=None):
    """
    Creates a bar chart comparing stay vs switch strategies.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing simulation results
    save_file : str, optional
        If provided, save the plot to this filename
    """
    strategies = ['Stay', 'Switch']
    win_rates = [results['stay_win_rate'], results['switch_win_rate']]
    
    # Create theoretical win rates for comparison
    num_doors = results['num_doors']
    theoretical_stay = 1/num_doors * 100
    theoretical_switch = (num_doors-1)/num_doors * 100
    theoretical_rates = [theoretical_stay, theoretical_switch]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(strategies))
    bar_width = 0.35
    
    # Plot bars
    empirical = ax.bar([i - bar_width/2 for i in x], win_rates, bar_width, 
                      label='Empirical Results', color='blue', alpha=0.7)
    theoretical = ax.bar([i + bar_width/2 for i in x], theoretical_rates, bar_width,
                        label='Theoretical Probability', color='red', alpha=0.7)
    
    # Add some text for labels, title and axes ticks
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(f'Monty Hall Problem: Stay vs. Switch ({results["num_trials"]} trials, {num_doors} doors)')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, 100)
    ax.legend()
    
    # Add value labels on the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(empirical)
    add_labels(theoretical)
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file)
    
    plt.show()

def run_comprehensive_test(trials_list=[100, 1000, 10000, 100000], doors_list=[3, 5, 10]):
    """
    Runs multiple simulations with different parameters to show how results converge.
    
    Parameters:
    -----------
    trials_list : list
        List of different trial counts to run
    doors_list : list
        List of different door counts to test
    """
    print("\n===== COMPREHENSIVE MONTY HALL TESTING =====")
    
    for num_doors in doors_list:
        print(f"\n--- TESTING WITH {num_doors} DOORS ---")
        print(f"Theoretical probabilities: Stay ({1/num_doors*100:.2f}%), Switch ({(num_doors-1)/num_doors*100:.2f}%)")
        
        for num_trials in trials_list:
            results = monty_hall_simulation(num_trials=num_trials, num_doors=num_doors)
            
            print(f"\nWith {num_trials} trials:")
            print(f"Stay: {results['stay_win_rate']:.2f}%, Switch: {results['switch_win_rate']:.2f}%")
            
            # Calculate how close the empirical results are to theoretical probabilities
            stay_theoretical = 1/num_doors * 100
            switch_theoretical = (num_doors-1)/num_doors * 100
            
            stay_diff = abs(results['stay_win_rate'] - stay_theoretical)
            switch_diff = abs(results['switch_win_rate'] - switch_theoretical)
            
            print(f"Deviation from theory: Stay: {stay_diff:.2f}%, Switch: {switch_diff:.2f}%")

# Example usage
if __name__ == "__main__":
    # Set parameters for the simulation
    NUM_TRIALS = 10000  # Number of games to simulate
    NUM_DOORS = 3       # Number of doors (classic version uses 3)
    DEBUG = True        # Show detailed information for first few games
    
    print("Running Monty Hall Problem Simulation...")
    print(f"Parameters: {NUM_TRIALS} trials, {NUM_DOORS} doors")
    
    # Run the simulation
    results = monty_hall_simulation(num_trials=NUM_TRIALS, num_doors=NUM_DOORS, debug=DEBUG)
    
    # Print and plot the results
    print_results(results)
    plot_results(results)
    
    # Run comprehensive testing with different parameters
    run_comprehensive_test()
    
    print("\nTry running the simulation with different parameters!")
    print("For example, try increasing the number of doors to see how it affects the win rates.")