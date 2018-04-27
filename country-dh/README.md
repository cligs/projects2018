# Readme
This folder contains detailed results for the paper
"A White-Box Model for Detecting Author Nationality by Linguistic Differences in Spanish Novels"
(Zehe, Albin ; Schlör, Daniel ; Henny-Krahmer, Ulrike ; Becker, Martin ; Hotho, Andreas In: DH, 2018).

* There is an iPython-Notebook for visualisation available as `visualise.ipynb`.
It provides information about the corpus as well as plots for the parameter study.
* The folder `output` contains the results for all tested configurations in our
parameter study.
The filenames in that folder follow the format
`country_segment-[NUMBER OF SEGMENTS]_C-[SVM COST PARAMETER]_removeuc-[REMOVE UPPERCASE TOKENS]_minn-[MINIMUM N-GRAM SIZE]_maxn-[MAXIMUM N-GRAM SIZE]_numfeatures-[NUMBER OF FEATURES]`.
* `requirements.txt` lists all python packages needed to run the iPython-Notebook.
