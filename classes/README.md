# classes: classes written in Python for creating our framework.

This directory contains all the classes that we wrote to create our framework. You do not need to run these
scripts directly. The main Python scripts uses these classes to perform the main tasks. So, keep these classes
unchanged.

* **Files and related classes:**
  * **data_tools:** Contains DataTools class which has all the methods related to IO and file types.
  * **engineer:** Contains Engineer class which has all the methods needed for performing weak supervision tasks using Snorkel.
  * **entity:** Contains EntityDetector class which is needed to do the entity detection task.
  * **labeling_functions:** Contains LabelingFunctions class. In this class, we implemented all the LFs.
  * **language_processing:** Contains LanguageProcessing class. In this class, we implemented all the tasks related to NLP using spaCy and other related libraries.
  * **preprocessing:** Contains Preprocessing class which performs the preprocessing tasks on the input text.
