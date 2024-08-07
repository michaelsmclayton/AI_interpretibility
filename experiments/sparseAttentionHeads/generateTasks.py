import numpy as np

# Create category words
categories = {
    "color": ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "black", "white"],
    "animal": ["cat", "dog", "bird", "wolf", "lion", "fish", "frog", "snake", "lizard", "monkey"],
    "fruit": ["apple", "pear", "banana", "cherry", "grape", "strawberry", "pineapple", "kiwi", "orange", "lemon"],
    "month": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"],
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "bodypart": ["head", "neck", "arm", "hand", "finger", "leg", "foot", "toe", "ear", "eye"]
}
category_names = list(categories.keys())

# Define function to generate prompt
def generatePrompt(n_examples, n_categories, n_words_per_prompt, n_words_per_category, scores, categories):
    # Get current categories
    cur_categories = np.random.choice(category_names, n_categories, replace=False)
    # Create score for each category
    category_scores = np.random.choice(scores, n_categories, replace=False)
    category_scores = {cat: category_scores[i] for i,cat in enumerate(cur_categories)}
    # Get current catagory words
    category_words = {category: list(np.random.choice(categories[category], n_words_per_category, replace=False)) for category in categories}
    # Generate prompts
    prompt = ""
    for i in range(n_examples):
        cats = np.random.choice(cur_categories, n_words_per_prompt, replace=False)
        words = [np.random.choice(category_words[cat]) for cat in cats]
        score = np.sum([category_scores[cat] for cat in cats])
        string = " ".join(words) + f": {score}" + "\n"
        prompt += string
    # Return prompt
    return prompt, category_scores

# Define current task parameters
n_examples = 30
n_categories = 5
n_words_per_prompt = 3
n_words_per_category = 1
scores = np.r_[0, np.arange(n_categories-1)]
prompt, category_scores = generatePrompt(n_examples, n_categories, n_words_per_prompt, n_words_per_category, scores, categories)
print(prompt)
print(category_scores)






