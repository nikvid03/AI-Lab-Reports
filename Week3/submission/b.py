import random

def generate_k_sat(k, m, n):
    # Validate input
    if k > n:
        raise ValueError("Number of literals per clause (k) cannot exceed number of variables (n).")

    # Initialize the list to store clauses
    clauses = []

    for _ in range(m):
        # Generate a clause with k distinct variables
        variables = random.sample(range(1, n + 1), k)  # Select k distinct variables
        clause = []
        
        for var in variables:
            # Randomly choose to negate the variable or not
            if random.choice([True, False]):
                clause.append(-var)  # Negate the variable
            else:
                clause.append(var)  # Use the variable as is

        clauses.append(clause)

    return clauses

def main():
    # Get user input for k, m, and n
    k = int(input("Enter the number of literals per clause (k): "))
    m = int(input("Enter the number of clauses (m): "))
    n = int(input("Enter the number of variables (n): "))

    try:
        # Generate the k-SAT problem
        k_sat_problem = generate_k_sat(k, m, n)

        # Print the generated k-SAT problem
        print("Generated k-SAT problem:")
        for clause in k_sat_problem:
            print(clause)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
