# prompts/qa_prompt_template.py

def build_qa_prompt(question, context):
	#
	prompt = f"""Answer the following question based on the provided context.
            ### Instructions:
            * Task: Identify the correct answer from the provided context.  
            * Approach:
            - Break down the problem into smaller parts, if necessary.
            - Carefully reason through your answer step-by-step.
            - Ensure that your answer is directly supported by the context.
            ### Response Format:
            Please format your answer within brackets as follows: 
			"<ans> Your Answer </ans>"
            ### 
            * Question: {question}
            * Context: {context}
		"""
	return prompt


