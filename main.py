# main.py - ReqCheck MVP Script
# This script checks a single requirement statement for ambiguous "weak" words.

# Phase 1: Establish Core Logic - The "Weak Word" List
# A list of common words and phrases that can lead to ambiguous requirements.
WEAK_WORDS = [
    "should", "may", "could", "possibly", "as appropriate", "user-friendly",
    "robust", "efficient", "effective", "etc.", "and/or", "minimize",
    "maximize", "support", "seamless", "easy to use", "state-of-the-art",
    "best", "handle", "approximately", "as required"
]

def check_requirement(requirement_text):
    """
    Analyzes a requirement string for weak words and returns the findings.
    
    Args:
        requirement_text (str): The requirement statement to check.
        
    Returns:
        list: A list of weak words found in the requirement.
    """
    # Create an empty list to store any weak words we find.
    found_words = []
    
    # Convert the input to lowercase to make the check case-insensitive.
    lower_requirement = requirement_text.lower()
    
    # Loop through our list of weak words.
    for word in WEAK_WORDS:
        # Check if the weak word is present in the requirement.
        if word in lower_requirement:
            found_words.append(word)
            
    return found_words

def main():
    """
    Main function to run the requirement checker.
    """
    print("--- Requirement Clarity Checker (ReqCheck MVP) ---")
    
    # Prompt the user to enter a requirement.
    user_input = input("Please enter a single requirement statement to check:\n> ")
    
    # Call the checking function with the user's input.
    ambiguous_words = check_requirement(user_input)
    
    # Print the results.
    if ambiguous_words:
        print("\n--- ANALYSIS ---")
        print(f"⚠️ Warning: Found {len(ambiguous_words)} potentially ambiguous word(s).")
        print(f"   Ambiguous words: {ambiguous_words}")
        print("   Consider replacing these words with more precise language.")
    else:
        print("\n--- ANALYSIS ---")
        print("✅ Requirement appears clear of common weak words.")

# This standard Python construct ensures the main() function is called when the script is run directly.
if __name__ == "__main__":
    main()