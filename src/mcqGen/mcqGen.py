import json
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
api_key = os.getenv("MY_KEY")

# Initialize the language model
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key, temperature=1.0)

# Define the desired JSON format
RESPONSE_JSON = {
    "questions": {
        "mcq": [
            {
                "question": "Multiple choice question",
                "options": [
                    { "a": "Option A" },
                    { "b": "Option B" },
                    { "c": "Option C" },
                    { "d": "Option D" }
                ],
                "correct_answer": "True Option"
            }
        ],
        "short_note": [
            {
                "question": "Short note question"
            }
        ],
        "blank": [
            {
                "question": "Blank question",
                "answer": "correct_answer"
            }
        ]
    }
}

# Define the prompt template
PROMPT_TEMPLATE = """
Text: {text}
You are an expert MCQ maker. Given the above text, create 5 multiple choice questions, 5 short note questions, and 5 fill-in-the-blank questions.
Ensure to format your response as follows:
{{"questions": {{"mcq": [{{"question": "Sample MCQ", "options": [{{"a": "Option A"}}, {{"b": "Option B"}}, {{"c": "Option C"}}, {{"d": "Option D"}}], "correct_answer": "a"}}], "short_note": [{{"question": "Sample short note question"}}], "blank": [{{"question": "Sample blank question", "answer": "correct_answer"}}]}}}}
"""

# Create the prompt template
prompt_template = PromptTemplate(
    input_variables=["text"],
    template=PROMPT_TEMPLATE
)

# Create the LLMChain
chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)

# Define the text for quiz generation
TEXT = "Tracks to Taj Nagar For twenty five years, residents of Taj Nagar village near Gurgaon lobbied for a railway station in their village. When their demand was not met, the villagers decided to take matters into their own hands. They pooled in twenty one lakh rupees and built a railway station on their own. Most of the three thousand people living in the village are farmers. But such was the burning desire to have a station in the village, everybody contributed according to their capacity. Ranging from three thousand rupees to seventy five thousand rupees. \"They donated money for the station and we started the construction in January 2008.\" said Ranjit Singh, a former village sarpanch. \"There are a large number of people in the village who need to go to Gurgaon, Delhi and Rewar. There are students who go to colleges. Till now, we had to either go to Halimandi or Patli to catch a train. Both the stations are six kilometers away from Taj Nagar. We thought when the railway lines passed through the village we would have a station here. But that didn\'t happen. So we raised the demand in 1982 and have been continuously asking for it, but the railways told us that they did not have funds. So, finally we decided to craft our own destiny,\" said Hukamchand, a member of the committee. As a result, the panchayat passed a resolution in 2008, saying that since the railway was not able to build a station for them, they would do it for themselves and with their own money! Soon, an eleven member team was formed and the team started collecting money from villagers. How would you solve these problems with minimum help from others? AGAINST THE ODDS On 7 January 2010, as a result of their efforts, the first railway station in the country on which the railway did not have to spend a single rupee, started operations. Sitapur\'s Light In rural Uttar Pradesh, over sixty percent of households are without power. Sitapur district is one such place with no power. A small social enterprise called Mera Gao Power (MGP) is trying to change things. They are putting two solar panels at a time. In just over a year, MGP has connected more than 3,500 customers to solar power mini- grids at a village level. Village by village, MGP is building a network of low cost solar micro-grids that provide two LED lights and a mobile charging point to all paying house holds at a cost of twenty five rupees per week. That is cheaper than kerosene which can cost almost double across a month. Solar power, as a \'smokeless\' source of light, comes with added benefits to customer health. Installing a micro-grid is a grand event in the village and every one gets involved. In the village of Damdampurawa, the team maps the village house by house beneath the scorching mid-day sun, working out where to place each wire so as to connect customer to the power source. Some house holders join in while others look on, calling out orders or watching the curious proceedings wide- eyed. The roof of a sturdy, brick-walled home in each village is always chosen as the site for the panels and the battery. Azaz, one of the company\'s first electrician to be recruited from the local district block of Reusa, installs English (S.L.), Std. 10 the panel in a southerly direction to capture as much sun light as possible. \"We\'re saving our environment with these lights, and there\'s no pollution in our homes either,\" says a farmer from the village. \"New businesses are starting to emerge amongst the customers too,\" says another. \"In one village, customers are using the light to weave saris by night. In another, one man now has a night business making plastic tablecloth,\" he says. \"Tt\'s nice to have light while we cook and eat. Our children are also studying more now!\" Palakkad\'s Public Library In kerala The Palakkad District Public Library has been up and running since September 2013. It is a fine modern library, a center for information, knowledge, wisdom, cultural activities, research and reference. But it has recently been in the news for different reasons. A third of its thousand members are women. These women, supported by the shared space the library offered them, launched a women\'s unit in February, 2014. The unit got together to discuss methods of empowering women. Glossary The library opened its halls for film screening, workshops in home economics or gardening, child care or the arts, and for women to get help in managing family conflicts, legal disputes and professional problems. The secretary of the library pointed out that through reading, women would realize their own strength and forge a unity. It was noted that the lending libraries of earlier times were disappearing and the present rural reading rooms were too often full of only male readers. The unit discussed that if the once well-read women of Kerala continued to squander their hours in front of television, it would encourage a climate in which women are afraid to go out after dark. So, the unit has formulated plans on opening separate reading rooms for women. Palakkad\'s district library stands tall asa beacon to encourage women\'s empowerment through classes, clubs, workshops and reading rooms. And then, there are the books, which will provide the women the strength they need to make good use of these opportunities."

# Generate the quiz
response = chain.invoke({"text": TEXT})

# Extract and clean the quiz string
quiz_str = response.get("text")
if quiz_str is None:
    print("No quiz string received.")
else:
    print("Raw quiz string:", quiz_str)
    quiz_str = quiz_str.strip("```json").strip("```") if quiz_str else "{}"
    print("Cleaned quiz string:", quiz_str)

# Validate and fix the JSON
def validate_json(json_str):
    try:
        return json.loads(json_str), None
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

# Validate the JSON
quiz_dict, error = validate_json(quiz_str)

if error:
    print(f"Error parsing JSON: {error}")
    quiz_dict = {"questions": {"mcq": [], "short_note": [], "blank": []}}
else:
    print("JSON is valid.")
    print("Parsed JSON:", quiz_dict)

# Prepare data for DataFrame
quiz_table_data = []
for question_type, questions in quiz_dict.get("questions", {}).items():
    if question_type == "mcq":
        for question in questions:
            mcq = question.get("question", "")
            options = " | ".join([f"{list(option.keys())[0]}: {list(option.values())[0]}" for option in question.get("options", [])])
            correct = question.get("correct_answer", "")
            quiz_table_data.append({
                "Question Type": "Multiple Choice",
                "Question": mcq,
                "Choices": options,
                "Correct Answer": correct
            })
    elif question_type == "short_note":
        for question in questions:
            short_note = question.get("question", "")
            quiz_table_data.append({
                "Question Type": "Short Note",
                "Question": short_note,
                "Choices": None,
                "Correct Answer": None
            })
    elif question_type == "blank":
        for question in questions:
            blank = question.get("question", "")
            correct = question.get("answer", "")
            quiz_table_data.append({
                "Question Type": "Fill in the Blank",
                "Question": blank,
                "Choices": None,
                "Correct Answer": correct
            })

# Convert to DataFrame
df = pd.DataFrame(quiz_table_data)
print(df)
