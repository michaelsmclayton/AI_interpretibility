import numpy as np

# Create category words
categories = [
    {"month": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"],
    "color": ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "black", "white"]},
    {"animal": ["cat", "dog", "bird", "wolf", "lion", "fish", "frog", "snake", "lizard", "monkey"],
    "fruit": ["apple", "pear", "banana", "cherry", "grape", "strawberry", "pineapple", "kiwi", "orange", "lemon"]},
    {"weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "bodypart": ["head", "neck", "arm", "hand", "finger", "leg", "foot", "toe", "ear", "eye"]},
    {"tool": ["hammer", "screwdriver", "saw", "drill", "pliers", "wrench", "screw", "nail", "bolt", "nut"],
     "vehicle": ["car", "bus", "train", "plane", "bike", "boat", "truck", "motorbike", "helicopter", "submarine"]},
    {"country": ["USA", "China", "Russia", "India", "Brazil", "Japan", "Germany", "UK", "France", "Italy"],
     "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin"]},
    {"sport": ["football", "basketball", "baseball", "soccer", "tennis", "golf", "hockey", "volleyball", "rugby", "cricket"],
     "music": ["rock", "pop", "hip-hop", "jazz", "classical", "blues", "country", "reggae", "metal", "folk"]},
    {"food": ["pizza", "burger", "pasta", "sushi", "steak", "salad", "soup", "sandwich", "taco", "burrito"],
     "drink": ["water", "juice", "soda", "beer", "wine", "coffee", "tea", "milk", "smoothie", "cocktail"]},
    {"clothing": ["shirt", "pants", "shoes", "hat", "jacket", "socks", "gloves", "scarf", "dress", "skirt"],
     "accessory": ["watch", "ring", "necklace", "bracelet", "earrings", "belt", "bag", "hat", "sunglasses", "tie"]},
    {"job": ["doctor", "teacher", "engineer", "scientist", "artist", "chef", "pilot", "nurse", "police", "firefighter"],
     "hobby": ["reading", "writing", "drawing", "painting", "cooking", "gardening", "photography", "knitting", "sewing", "woodworking"]},
]

# Define categories
category_names = ["month", "color", "animal", "fruit", "weekday", "bodypart", "tool", "vehicle", "country", "city", "sport", "music", "food", "drink", "clothing", "accessory", "job", "hobby", "object", "appliance"]
n_categories = len(category_names)

# Define category order
category_order = [list(c.keys()) for c in categories]

# Defie category scores
scoreValues = np.array([list(np.random.choice([0,1], 2, replace=False)) for i in range(len(category_order)+1)]).flatten()
# scores = {category: scoreValues[i] for i,category in enumerate(category_names)}
scores = {category: scoreValues[c] for c,category in enumerate(category_names)}

# Generate task example
def makeExample(n_words_per_prompt, n_words_per_category, scores):
    # Get words and scores
    cur_words = []; cur_scores = []
    for n in range(n_words_per_prompt):
        cur_category = np.random.choice(category_order[n])
        cur_words.append(np.random.choice(np.array(categories[n][cur_category])[:n_words_per_category]))
        cur_scores.append(scores[cur_category])
    # Generate example
    prompt = " ".join(cur_words) + ": " + str(np.sum(cur_scores))
    return prompt

# Define function to generate prompt
def generateTaskExamples(n_examples, n_words_per_prompt, n_words_per_category):
    # Generate prompts
    taskExamples = ""
    for i in range(n_examples):
        string = makeExample(n_words_per_prompt, n_words_per_category, scores) + "\n"
        taskExamples += string
    # Return prompt
    return taskExamples

# Shuffle all numbers in string
def shuffleNumbersInString(taskExamples):
    s = np.array([i for i in taskExamples])
    # Get all numbers in string
    idx, num = np.array([[i,num] for i,num in enumerate(s) if num.isdigit()]).astype(int).T
    # Shuffle numbers
    num_shuffled = np.random.permutation(num)
    # Replace numbers in string
    s[idx] = num_shuffled
    # Return string
    return "".join(s)

# Split task examples by ":"
def getPromptAndAnswer(t):
    t_split = t.split(":")
    prompt = ":".join(t_split[:-1]) + ":"
    answer = t_split[-1].split("\n")[0]
    return prompt, answer