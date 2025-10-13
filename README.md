# Deliverables 

## Intermediate submission 

1. By **Sunday November 10**, you should have chosen a team and a topic.
2. On **Friday November 22** your team will submit a 1-2 page writeup. Your writeup should provide a preliminary introduction to the topic you will study and provide clear motivation for why they are interesting and/or relevant. 

## Final submission 

You are required to hand in a PDF version of your report `report.pdf` (**max 20 pages**) and the source code used. You should not show the actual code in the PDF report, unless you want to point out something specific.

Your `README.md` should contain instructions on reproducing the PDF report from the quarto file. This can be useful if you have issues with the automatic generation of the PDF report right before the deadline. Your `README.md` should also include a brief description of the contributions of each team member, if you are a team of three students.

**Checklist**:

1. [ ] `report.pdf` in GitHub repository (e.g., generated from `report.qmd` or `report.tex`) (**max 20 pages**)
2. [ ]  source code in GitHub repository (should be able to run from top to bottom)
3. [ ] `README.md` with instructions on how to run the code, reproduce the PDF report, and a brief description of the team members' contributions, if applicable (please delete everything not related to your project in it)


# Project

The goal of this project is quite broad, students are free to come up with their own ideas. While simulation studies are the designated topic, groups that found interesting data during the small project and would like to carry on analyzing it, or groups interested in studying a bit deeper one of the methodological concepts from this course are encouraged to approach the teachers during the exercises and discuss their ideas. **Prospective topics for the final project will be gradually revealed during the lectures.**

Part of the grade for the final project (10 % of the total grade, i.e., one quarter of the final project) will be awarded for value added (original data analysis, simulation study answering a previously unclear question, etc.). All of the prospective topics that will be introduced during the lecture will have this element, and by half-way through the semester (when the final project will start) it should be clear through the examples what the project should aspire to. We will also discuss this in person at some point, likely on Week 7. The remaining three quarters of the project (i.e., 30 % of the total grade) will be awarded for

- quality of the report _(clarity, readability, structure, referencing, etc.)_
- graphical considerations _(well chosen (as discussed during the course) graphics with captions, referenced from the main text)_
- concepts explored beyond the scope of the course _(in the soft sense that they were not fully covered during classes)_
- overall quality _(correctness, demonstration of understanding, etc.)_

A project seriously lacking in any of the criteria above will be penalized.


## Topics for the final project


_Unless otherwise stated, you are not required to code everything from scratch (unless there is no available source code)._
  

### 1. Cross-validation for PCA

   - A simulation study to compare the advantages of EM over the repaired CV for PCA, covered in [Week 4](https://math-516-517-main.github.io/math_517_website/lectures/05_CV.pdf).
   - Implement a third approach to CV for PCA based on matrix completion (boils down to performing SVD with missing data). Details about this approach are given in [the supplementary notes](https://math-516-517-main.github.io/math_517_website/notes/week_05.html) and a deeper dive into the matter is covered in [Perry (2009), Section 5](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=0a2508a9fd89513ddf1c227274c8192993520692).
   - Compare the three methods.

### 2. Comparison of variable selectors in regression

   - [Hastie et al. (2020)](https://projecteuclid.org/journals/statistical-science/volume-35/issue-4/Best-Subset-Forward-Stepwise-or-Lasso-Analysis-and-Recommendations-Based/10.1214/19-STS733.short) have some surprising results in their simulation study, but one important method (adaptive lasso) is omitted. Try to recreate the study with adaptive lasso included (and perhaps elastic net, too?). 
   - The project should address the following question: When it comes to variable selection, which method to choose and under which settings and for which aim/criterion (explainability, predictability, sparsity, or number of correct covariates)?

### 3. Comparison of cross-validation methods for data with temporal structure

When short-range temporal dependence (autocorrelation) is not taken care of, "simple" cross-validation methods can break down as the validation and training samples are no longer independent. For instance, the "simple" CV approaches can lead to underestimation of smoothing parameters (overfitting). Several methods for dealing with such issues have been proposed in the literature. Among these, we mention the following
-  removing distance-based buffers around hold-out points in the LOOCV
-  block cross-validation
-  neighborhood cross validation, recently proposed by [Wood (2024)](https://arxiv.org/pdf/2404.16490v2)

The project aims at 

 - exploring why cross-validation might fail in presence of temporal (or spatial) dependence, and
 - making a survey of some of the modified cross-validation methods proposed in the literature to deal with autocorrelation and to compare them in a non-parametric regression setting (e.g., with smoothing splines).
 
Naturally, different short-range autocorrelation schemes should be investigated (e.g., AR or MA Gaussian processes). 

References related to this subject: [Chu and Maaron (1991)](https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-4/Comparison-of-Two-Bandwidth-Selectors-with-Dependent-Errors/10.1214/aos/1176348377.full), [Arlot and Celisse (2010)](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://projecteuclid.org/journals/statistics-surveys/volume-4/issue-none/A-survey-of-cross-validation-procedures-for-model-selection/10.1214/09-SS054.pdf&ved=2ahUKEwiqjbDdkKSJAxX5if0HHd4mPZAQFnoECA4QAQ&usg=AOvVaw2z3xj4V6JzRULYBwGScd-l), [Roberts et al. (2016)](https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881).

### 4. The EM algorithm for different patterns of missingness

- Comparison of the performances of the EM algorithm for different percentage of missing values and for different missing-data mechanisms.
- *Optional (if you have time and energy):* In a setting with missing data, what about comparing parameter estimates obtained via EM with those obtained (with maximum likelihood) after imputation? You can choose one (or more) of the imputation methods described [here](https://rmisstastic.netlify.app/how-to/impute/missImp.pdf).
- The project should address the following question: When you are faced with a missing data problem, when is the EM algorithm a good option for statistical inference (the estimation process)?

### 5. Diving into one of the course topics, e.g., MM algorithms or Monte Carlo integration (Week 7)

Consult with the teacher to define the project and ensure its feasibility.

