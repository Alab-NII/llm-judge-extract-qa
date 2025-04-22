# LLM-as-a-Judge for Extractive QA Datasets 

This is the repository for the paper: [LLM-as-a-Judge: Reassessing the Performance of LLMs in Extractive QA](https://arxiv.org/abs/2504.11972)



## Reproduction of Results
- download [data.zip](https://www.dropbox.com/scl/fi/pmxaxt5ibj9mlzui5gbpw/data.zip?rlkey=dc2nuie1ad47kkk7ta8slfklk&st=6pycuere&dl=0)
- download [judge.zip](https://www.dropbox.com/scl/fi/5i88veb30xqbbj7bn0w71/judge.zip?rlkey=xobwku0k5m70vw74y3ssmztih&st=8n8eyb9b&dl=0) 

### Table 2: Pearson correlation coefficients
```bash
python3 get_correlation_score.py
```

### Table 3: Automatic evaluation scores (EM and F1) and LLM-as-a-judge scores
```bash
python3 run_eval.py
```



## Running process

### Step 1: Run the QA Task
```bash
python3 run_qa.py
```

#### Postprocessing the generated response from the QA task
```bash
python3 postprocess_qa.py
```

### Step 2: Run LLM-as-a-judge
```bash
python3 run_judge.py
```

### Step 3: Evaluation
```bash
python3 run_eval.py
```

#### Calculate correlation scores:
```bash
python3 get_correlation_score.py
```

## Data files include:
- [data](https://www.dropbox.com/scl/fi/pmxaxt5ibj9mlzui5gbpw/data.zip?rlkey=dc2nuie1ad47kkk7ta8slfklk&st=6pycuere&dl=0): input data
- data/human_result.json: human judgement dataset
- [qa_inference](https://www.dropbox.com/scl/fi/fe56j1m358tppcb14mjqx/qa_inference.zip?rlkey=15wdi626sebivuhu80b63pae2&st=v5cbz03q&dl=0): predicted answers from 8 QA models
- [qa_postprocess](https://www.dropbox.com/scl/fi/d8b8gljd8gvicm35lyzhr/qa_postprocess.zip?rlkey=w3nikidu7qg7zasdnobqshcli&st=3qrtm9b2&dl=0): postprocess on the predicted answers
- [judge/mistral-v0.3](https://www.dropbox.com/scl/fi/5i88veb30xqbbj7bn0w71/judge.zip?rlkey=xobwku0k5m70vw74y3ssmztih&st=8n8eyb9b&dl=0): judged by mistral-v0.3
- [judge/llama-3.3-70b](https://www.dropbox.com/scl/fi/5i88veb30xqbbj7bn0w71/judge.zip?rlkey=xobwku0k5m70vw74y3ssmztih&st=8n8eyb9b&dl=0): judged by llama-3.3-70b
- [judge/qwen-2.5-72b](https://www.dropbox.com/scl/fi/5i88veb30xqbbj7bn0w71/judge.zip?rlkey=xobwku0k5m70vw74y3ssmztih&st=8n8eyb9b&dl=0): judged by qwen-2.5-72b






