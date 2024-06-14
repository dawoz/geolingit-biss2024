### Bertinoro International Spring School 2024 Exam

# MINERVA VS Camoscio VS LLaMAntino

### **Authors**: [Davide Savarro] [Davide Zago] [Stefano Zoia] 

This repository includes some code to fine-tune, and test three different model: MINERVA, Camoscio (actually the ExtremITA modified version) and ANITA.

The repository contains the following files:

- [fine-tuning notebook](Exam_BISS-2024_LLM_finetuning.ipynb): code for fine-tuning the models. In the file ```secret.txt``` you have to insert your HuggingFace Access Token to at least access the original MINERVA model and tokenizer (after accepting the conditions on the model card) before starting the fine-tuning phase.
- [inference notebook](Exam_BISS-2024_LLM_inference.ipynb): code for inference and test for each model.
- [fine-tuning script](run_finetuning.py): python script export of the [fine-tuning notebook](Exam_BISS-2024_LLM_finetuning.ipynb) to run it in non interactive shells.
- [inference script](run_inference.py): python script export of the [inference notebook](Exam_BISS-2024_LLM_inference.ipynb) to run it in non interactive shells. Note: it exports the predictions of each model on the test set, and these are then used in the error analysis.
- [dataset analysis notebook](Exam_BISS-2024_LLM_dataset_analysis.ipynb): code for the dataset analysis.
- [error analysis notebook](Exam_BISS-2024_LLM_error_analysis.ipynb): code for the error analysis.

Due to file size, the notebook for dataset and error analysis saves the plots to file instead of showing them. You can uncomment the command that shows the image (e.g. `fig.show()`, `plt.show()`, ...).