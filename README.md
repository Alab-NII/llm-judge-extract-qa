# LLM-as-a-Judge for Extractive QA Datasets 

This is the repository for the paper: [LLM-as-a-Judge: Reassessing the Performance of LLMs in Extractive QA]()

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
- [mistral-v0.3](https://www.dropbox.com/scl/fi/h64voic4glt7bfrwrq60l/mistral-v0.3.zip?rlkey=xjfl8nkhzu8gab0qmwyz389t8&st=sl8kfiay&dl=0): judged by mistral-v0.3
- [llama-3.3-70b](https://www.dropbox.com/scl/fi/7extdn545nlu68trkb1fg/llama-3.3-70b.zip?rlkey=j2nlt74h0c54vrrmndkhoxl2g&st=ufghdtea&dl=0): judged by llama-3.3-70b
- [qwen-2.5-72b](https://www.dropbox.com/scl/fi/73pamu4rnc1846jue3z97/qwen-2.5-72b.zip?rlkey=uk04lcoyj96oq5sdara7zx94i&st=zx0relcz&dl=0): judged by qwen-2.5-72b







