# prompts/judge_prompt_template.py
import json

def format_examples(examples):
    formatted_examples = []
    #
    for idx, example in enumerate(examples, start=1):
        # Extracting values from the dictionary
        question = example.get("question")
        gold_answer = example.get("answer")
        predicted_answer = example.get("generated_ans")
        label = example.get("label")
        #
        # Formatting the example
        formatted_example = f"### Example {idx}:\n"
        formatted_example += f"  * Question: {question}\n"
        formatted_example += f"  * Gold Answer: {gold_answer}\n"
        formatted_example += f"  * Predicted Answer: {predicted_answer}\n"
        formatted_example += f"  * Label: <ans> {label} </ans>\n"
        #
        formatted_examples.append(formatted_example)
    #
    return "\n".join(formatted_examples)


with open("data/demonstrations.json", "r", encoding="utf-8") as f:
    demonstrations = json.load(f)

examplar_format = format_examples(demonstrations)


def build_judge_prompt(ques, gold_ans, pred_ans, context):
	prompt = f"""Your job is to evaluate a predicted answer by comparing it against the gold answer and the given question. 
                You may refer to the provided context if needed.
            ## Grading Criteria:
            * "CORRECT": The predicted answer matches the gold answer or is a valid alternative (e.g., different but correct ways of writing a name).
            * "INCORRECT": The predicted answer is wrong or does not align with the gold answer.
			* In some ambiguous cases, where it is unclear whether the predicted answer is correct or not, please refer to the provided context and use it as the final source for making your judgment.
			## Response Format:
            Please format your answer within brackets as follows: <ans> Your Answer </ans>.
            ## Here are some examples:
            {examplar_format}
            ## Here is your task:
            * Question: {ques}
            * Gold Answer: {gold_ans}
            * Predicted Answer: {pred_ans}
            * Context: {context}
            * <ans> YOUR LABEL </ans>
		"""
	#
	return prompt