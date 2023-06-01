<br />
<p align="center">
  <a href="https://dadebias.github.io/genda-lens/">
    <img src="docs/img/logo2.png" alt="Logo" width=350 height=150>
  </a>
  <h1 align="center">The GenDa Lens</h1> 
  <h3 align="center">Quantifying Gender Bias in Danish language models</h3> 
  <p align="center">
    Thea Rolskov Sloth & Astrid Sletten Rybner
    <br />
    <a 
    a>
    <br />
  </p>
</p>

A python package for investigating gender bias in Danish language models within the following domains:  

* **Language Modeling** (for pre-trained models)  

* **Coreference Resolution** (for coref. models)  

* **Named Entity Recogntiion** (for NER models)  

----------

If you want to test either a pre-trained model, a coref. model or a NER model, you can read more about each of these three types of tests in the User Guide.  

Here you can also find a section on the *defintions* of harm, gender and bias that we adopt in the GenDa Lens package. 

# 🔎 [Documentation](https://dadebias.github.io/genda-lens/)
| Documentation          |                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| 📚 **[User Guide]**    | Instructions on how to understand the implemented Gender Bias tests                              |
| 💡 **[Definitions]**   | Defintions of harm, bias and gender applied in GenDa Lens                                        |
| 💻 **[API References]**| The detailed reference for the GenDa lens API. Including function documentation                  |
| 🧐 **[About]**         | Learn more about how this project came about and who is behind the implemented frameworks        |


[User Guide]: https://dadebias.github.io/genda-lens/user_guide/lm/
[Definitions]: https://dadebias.github.io/genda-lens/user_guide/metrics/
[About]: https://dadebias.github.io/genda-lens/about/
[API References]: https://dadebias.github.io/genda-lens/api/

## 🤗 Integration
Note that for NER and Language Modeling, the GenDa Lens evaluator is integrated with Hugging Face.

## 🔧 Installation
You can install GenDa Lens via pip from PyPI:

```bash
pip install genda_lens
```

## 👩‍💻 Usage
You can test your model by instatiating an instance of the Evaluator and running the appriate evaluation function:  

```python
from genda_lens import Evaluator

# initiate evaluator
ev = Evaluator(model_name="huggingface-modelname")

# run test
output = ev.evaluate_ner(n=20)

# retrieve output
simple_output = output[0]
detailed_output = output[1]
```

Subsequently, the output can be visualized using the Visualizer: 

```python
from genda_lens import Visualizer

# initiate visualizer
viz = Visualizer()

# visualize ner results
plot = viz.visualize_results(data = detailed_output_ner, framework = "ner", model_name "my-model-name")

```

### Acknowledgements
This project uses code from three already implemented frameworks for quantifying gender bias in Danish. 
While all code written by others is properly attributed at the top of the scripts in the repository, we would also like to present aknowledgement here to the authors of the work we draw on:

* *The original ABC Framework:*
[González, A. V., Barrett, M., Hvingelby, R., Webster, K., & Søgaard, A. (2020). Type B reflexivization as an unambiguous testbed for multilingual multi-task gender bias.](https://arxiv.org/pdf/2009.11982.pdf) 

* *The original Augmented DaNe Framework:*
[Lassen, I. M., Almasi, M., Enevoldsen, K., & Kristensen-mclachlan, R. (2023, May). Detecting intersectionality in NER models: A data-driven approach.](https://aclanthology.org/2023.latechclfl-1.13.pdf) 

* *The original WinoBias Framework:*
[Zhao, J., Wang, T., Yatskar, M., Ordonez, V., & Chang, K. W. (2018). Gender bias in coreference resolution: Evaluation and debiasing methods. ](https://arxiv.org/pdf/1904.03310.pdf) 

* *The Danish translation of the WinoBias Framework, DaWinoBias:*
[Signe Kirk and Kiri Koppelgaard](https://github.com/NLP-exam/DaWinoBias) 

