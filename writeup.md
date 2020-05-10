# Title

Team: JHU-CLSP 

- Marie-Philippe Gill
- Nanxin Chen
- Saurabhchand Bhati
- Sonal Joshi 
- Laureano Moro-Velazquez


Affiliations

# Summary Sentence
```
(suggested limit 30 words)

Include 1 short sentence description of your method that can be used in a main text table.
```

We used `tsfresh` to extract timeseries features followed by xgboost with hyperparameter search to predict the on/off status, and the severity of tremor & dyskinesia.

# Background/Introduction

```
(suggested limit 200 words)

Please try to address the following points:

- What is the motivation for your approach?
  This will include any previous work and observations that you have made about the data
  to suggest your approach is a good one. Provide the reader with an intuition of how you
  approached the problem
- What is the underlying methodology used (e.g., SVM or regression)?
- Where there any novel approaches taken in regards to feature selection, data imputation, ranking, etc?
```

For the duration of the challenge, we were exploring in parallel two approaches: using the tsfresh package to extract features and giving those features to an xgboost, and the second approach is using a Support Vector Regressor (SVR) in a couple of ways. 

- `SVR`: one SVR per subject, sharing the same hyperparameters

- `SVR Per Patient`: one SVR per subject, each model have their own hyperparameters tuned. Our motivation to try this was because we realized the MSE per patient was varying a lot. We thought that choosing the best hyperparameters might lead to overfitting, but it was worth trying considering the data might be very different from one subject to another as labels are very subjective to the subject's perception of the severity of their symptoms.

- `SVR Everyone`: one SVR for all the data, and we added subject information with a one-hot vector. 

# Methods

```
(suggested limit 800 words)

The methods should cover a full description of your methods so a reader can reproduce them.
Please cover how you processed the data, if any data was imputed or manipulated in any way
(e.g., you mapped data onto pathways or combined different datasets), the underlying algorithm,
any modifications to the underlying method of importance, the incorporation of outside data, and
the approach to predict submitted data.

If you submitted multiple predictions, please specify which is the difference among them
(e.g. only parameters tuning or different algorithms). If needed, you can decide to write one
sub-paragraph for each submission.
```

## Data Preparation  

[[/images/features-diagram.png|Different input data used]]

*Figure ?: Different data we used as input.*

When we received the data and plotted it in a graph, it was obvious that some measurements had very straight lines that looked like the subject had to remove their watch as they were completely still. Also, the signal wasn't centered around zero. Those inactive lines could be higher than 0, so a high pass filter was applied to correct it. It was then possible to detect inactivity and remove it.

To identify and remove inactivity, there are two conditions:

- A value needs to be lower than the energy threshold. If `energy_threshold = 10`, then the threshold for each axis will be 10% of that max value. Candidates to inactivity removal become every value that's lower than that threshold. 

- The second condition, called `duration_threshold`, represents the minimum number of consecutive candidates to be removed before the candidates will be indeed removed and confirmed as inactivity. For example, we could decide to only remove sections that are at least 1 minute long of inactivity detected. If it's lower, we don't consider it as inactivity as it's not long enough and we keep the values. 


For tsfresh and xgboost, the data used was always the original one where inactivity was not removed. Some experiments were made when inactivity was removed, but they didn't provide better results. However, we didn't perform a gridsearch on the inactivity removed data, so maybe that could provide some improvements. 

For the SVR, the best results were obtained on original data and inactivity removed for ON/OFF and tremor, but dyskinesia was obtaining better results with just original data where inactivity was not removed. 

For all the submissions, we focused our work on the CIS-PD database. 

## Cross-validation with 5-folds 

We empirically decided to do a 5-fold cross-validation per patient. All the measurements of a patient were divided into 5 groups, where we tried to make the groups as balanced as possible, using only on/off as the unique label to balance the groups. In CIS-PD, the subject `1046` did not have any labels for on/off, so tremor was used for that subject. 

Creating folds on a multilabel dataset certainly proved to be a challenge, where there might still be some room for improvement.

## Intermediate submissions 


<table>
  <tr>
    <th>Submission</th>
    <th><span style="font-weight:bold">Subchallenge</span></th>
    <th><span style="font-weight:bold">Scheme</span></th>
    <th><span style="font-weight:bold">Fusion</span></th>
    <th>Final Score<br>on Test Folds</th>
    <th>Ranking</th>
  </tr>
  <tr>
    <td rowspan="3">1</td>
    <td>ON/OFF</td>
    <td>tsfresh</td>
    <td></td>
    <td></td>
    <td>6/15</td>
  </tr>
  <tr>
    <td>Tremor</td>
    <td>tsfresh</td>
    <td></td>
    <td></td>
    <td>9/25</td>
  </tr>
  <tr>
    <td>Dyskinesia</td>
    <td>tsfresh</td>
    <td></td>
    <td></td>
    <td>8/13</td>
  </tr>
  <tr>
    <td rowspan="3">2</td>
    <td>ON/OFF</td>
    <td><span style="font-weight:400;font-style:normal">tsfresh with global normalization</span></td>
    <td></td>
    <td></td>
    <td>3/27</td>
  </tr>
  <tr>
    <td>Tremor</td>
    <td><span style="font-weight:400;font-style:normal">tsfresh with global normalization + SVR</span></td>
    <td><span style="font-weight:400;font-style:normal">Gradient Boosting Regression</span></td>
    <td></td>
    <td>9/25</td>
  </tr>
  <tr>
    <td>Dyskinesia</td>
    <td><span style="font-weight:400;font-style:normal">tsfresh with global normalization + SVR</span></td>
    <td><span style="font-weight:400;font-style:normal">Gradient Boosting Regression</span></td>
    <td></td>
    <td>8/25</td>
  </tr>
  <tr>
    <td rowspan="3">3</td>
    <td>ON/OFF</td>
    <td><span style="font-weight:400;font-style:normal">tsfresh with hyperparameter search</span></td>
    <td></td>
    <td></td>
    <td>4/29</td>
  </tr>
  <tr>
    <td>Tremor</td>
    <td><span style="font-weight:400;font-style:normal">tsfresh with hyperparameter search</span></td>
    <td></td>
    <td></td>
    <td>2/27</td>
  </tr>
  <tr>
    <td>Dyskinesia</td>
    <td><span style="font-weight:400;font-style:normal">tsfresh with hyperparameter search</span></td>
    <td></td>
    <td></td>
    <td>3/27</td>
  </tr>
  <tr>
    <td rowspan="3">4</td>
    <td>ON/OFF</td>
    <td><span style="font-weight:400;font-style:normal">tsfresh with per-patient tunning and using training data for stop criterion</span></td>
    <td></td>
    <td></td>
    <td>4/34</td>
  </tr>
  <tr>
    <td>Tremor</td>
    <td><span style="font-weight:400;font-style:normal">tsfresh with per-patient tunning and using training data for stop criterion</span></td>
    <td></td>
    <td></td>
    <td>5/33</td>
  </tr>
  <tr>
    <td>Dyskinesia</td>
    <td><span style="font-weight:400;font-style:normal">tsfresh with per-patient tunning and using training data for stop criterion</span><br><span style="font-weight:400;font-style:normal"> + i-vectors with SVR per-patient tunning</span></td>
    <td><span style="font-weight:400;font-style:normal">average of predictions</span></td>
    <td></td>
    <td>3/32</td>
  </tr>
</table>

*Table ?: Summary of all the experiments for the 4 intermediate rounds of submission.* 


| Submission  | Series | Features
| ------------- | ------------- |  ---------- |
| 1  | x, y, z  | numpy(mean,max,min,std,var,ptp,percentile(10,20,30,40,50,60,70,80,90)), <br>scipy(skew,kurtosis,kstat(1,2,3,4), <br>moment(1,2,3,4),<br>tsfresh
| 2  | x, y, z  | 
| 3  | x, y, z, abs(delta_x), abs(delta_y), abs(delta_z)| +tsfresh(fft(4-34))
| 4  | x, y, z, abs(delta_x), abs(delta_y), abs(delta_z)|

*Table ?: Features used for CIS-PD for the different submissions*


### Submission 1 

For the first submission, `tsfresh` was used to submit the predictions for all the subchallenges. The hyperparameters were tuned manually. The best hyperparameters chosen for that submission are shown in table ?. 

For the REAL-PD database, the hyperparameters used are the ones we found that worked best on CIS-PD.


| Hyperparameters  	| Values           	|
|------------------	|------------------	|
| objective        	| reg:squarederror 	|
| max_depth        	| 3                	|
| learning_rate    	| 0.05             	|
| subsample        	| 0.8              	|
| colsample_bytree 	| 0.9              	|
| min_child_weight 	| 1                	|
| nthread          	| 12               	|
| random_state     	| 42               	|

*Table ?: Manual hyperparameters chosen for submission 1 & 2.*

### Submission 2  

For the second submission, we added global normalization for all subchallenges. For tremor and dyskinesia, a fusion of the predictions from tsfresh + xgboost and SVR were made using gradient boosting regression. Figure ? and ? respectively show the pipeline for tremor and dyskinesia. 

The parameters used are the same as for the first submission, and they are shown in table ?.

[[/images/submission4-pipeline-sub2-tremor.png|Pipeline for tremor in the 2nd submission]]

*Figure ?: Pipeline for tremor in the 2nd submission*

[[/images/submission4-pipeline-sub2-dysk.png|Pipeline for dyskinesia in the 2nd submission]]

*Figure ?: Pipeline for dyskinesia in the 2nd submission*

For the first two submissions, we found the best hyperparameters with CIS-PD database and we used the same ones for REAL-PD. 

For the REAL-PD database, we used the same hyperparameters. However, global normalization was nor performed like it was the case for CIS-PD in this second submission. 

### Submission 3

For the third submission, tsfresh with a hyperparameter search was submitted. At this time, the improvements obtained with the hyperparameter search were much better than the results obtained with the SVR so no fusion was performed. 

Some new features were added: 
- Fast Fourier Transform (FFT) 
- Features extracted from the absolute value were added (motivation was 2017 challenge paper)

🟡TODO: Add reference to that paper that was doing the absolute value thing 

For the REAL-PD database, the same prediction CSV file as submission 2 was used. One pitfall we encountered at this stage is that if we used the best set of hyperparameters found on the CIS-PD database, the exact same label was obtained for all the measurement ids of a subject, so that forced us to reuse the hyperparameters from submission 1 and 2. However, the new features were added.

<table>
  <tr>
    <th colspan="2"><span style="font-weight:400;font-style:normal">Hyperparameters</span></th>
    <th>Values</th>
  </tr>
  <tr>
    <td colspan="2">objective</td>
    <td><b>reg:squarederror</b></td>
  </tr>
  <tr>
    <td colspan="2">silent</td>
    <td><b>False</b></td>
  </tr>
  <tr>
    <td colspan="2">max_depth</td>
    <td><b>2</b>, 3, 4, 5, 6</td>
  </tr>
  <tr>
    <td colspan="2">learning_rate</td>
    <td>0.001, 0.01, 0.05, 0.1, 0.2, <b>0.3</b></td>
  </tr>
  <tr>
    <td colspan="2">subsample</td>
    <td>0.5, 0.6, 0.7, 0.8, 0.9, <b>1.0</b></td>
  </tr>
  <tr>
    <td colspan="2">colsample_bytree</td>
    <td>0.4, 0.5, 0.6, 0.7, <b>0.8</b>, 0.9, 1.0</td>
  </tr>
  <tr>
    <td colspan="2">colsample_bylevel</td>
    <td>0.4, <b>0.5</b>, 0.6, 0.7, 0.8, 0.9, 1.0</td>
  </tr>
  <tr>
    <td colspan="2">min_child_weight</td>
    <td><b>0.5</b>, 1.0, 3.0, 5.0, 7.0, 10.0</td>
  </tr>
  <tr>
    <td colspan="2">gamma</td>
    <td>0, 0.25, 0.5, <b>1.0</b></td>
  </tr>
  <tr>
    <td colspan="2">reg_lambda</td>
    <td>0.1, 1.0, 5.0, 10.0, 50.0, <b>100.0</b></td>
  </tr>
  <tr>
    <td colspan="2">n_estimators</td>
    <td>50, <b>100</b>, 500, 1000</td>
  </tr>
</table>

*Table ?: Gridsearch performed for the hyperparameters of the xgboost for the third submission. The values in bold are the ones that provided the best results on the CIS-PD database.* 


### Submission 4 

[[/images/submission4-pipeline-sub4-onoff-tremor.png|Pipeline for dyskinesia in the 4th submission]]

*Figure ?: Pipeline of the 4th submission for ON/OFF and tremor.*

`ON/OFF:` tsfresh with per-patient tunning and using training data for stop criterion

`Tremor:` tsfresh with per-patient tunning and using training data for stop criterion

`Dyskinesia:` average of predictions:
- tsfresh with per-patient tunning and using training data for stop criterion
- i-vectors with SVR per-patient tunning

🟡TODO: Explain what is the different between tsfresh and tsfresh with per-patient tuning 

🟡TODO: Explain what is training data for stop criterion 

[[/images/submission4-pipeline-sub4-dysk.png|Pipeline for dyskinesia in the 4th submission]]

*Fig ?: Pipeline of the 4th submission for dyskinesia.*

The ivector used for this 4th submission was extracted/trained(🧐?) on training original data and was of dimension 650. The autoencoder was with 60 features and a frame length of 400. We experimented with different values as shown in table (🙋‍♀️ref), but these provided the best results.

🧐QUESTION: Not sure if "Hyperparameters is a good word as I'm also talking about data?" 

<table>
  <tr>
    <th></th>
    <th colspan="2"><span style="font-weight:400;font-style:normal">Hyperparameters</span></th>
    <th>Values</th>
  </tr>
  <tr>
    <td>Input</td>
    <td colspan="2">Data</td>
    <td>Original training data, original training data + inactivity removed<br></td>
  </tr>
  <tr>
    <td>ivector</td>
    <td colspan="2"><span style="font-weight:400;font-style:normal">Dimension</span></td>
    <td>350, 400, 450, 500, 550, 600, 650, 700</td>
  </tr>
  <tr>
    <td rowspan="2">Autoencoder</td>
    <td colspan="2"><span style="font-weight:400;font-style:normal">Nb features</span></td>
    <td>30, 60</td>
  </tr>
  <tr>
    <td colspan="2">Framelength</td>
    <td>240, 320, 400, 480</td>
  </tr>
  <tr>
    <td rowspan="5">SVR</td>
    <td><span style="font-weight:400;font-style:normal">PCA</span></td>
    <td>Nb Components</td>
    <td>350, 400, 450, 500, 550, 600, 650, 700</td>
  </tr>
  <tr>
    <td colspan="2">Kernel</td>
    <td>linear</td>
  </tr>
  <tr>
    <td colspan="2"><span style="font-weight:400;font-style:normal">Epsilon</span></td>
    <td>0.1</td>
  </tr>
  <tr>
    <td colspan="2">C</td>
    <td>2E-13, 2E-11, 2E-9, 2E-7, 2E-5, 2E-3, 2E-1, 2E1 </td>
  </tr>
  <tr>
    <td colspan="2">Gamma</td>
    <td>auto</td>
  </tr>
</table>

*Table ?: Gridsearch performed on the hyperparameters for the i-vector and SVR*

The SVR Per-Patient tuning predictions are made by training one SVR per subject. However, the hyperparameters for the SVR are tuned for each subject individually, so they have different values for the `C` parameter of the SVR and also a different number of components when PCA is performed. 

To find which configuration is the best for each subject, we compute the weighted final score over the 5 folds for each subject. Then, we choose the configuration that has the lowest weighted final score as the best configuration. 

Initially, we made some experiments with different kernels and epsilon values, but the best results were obtained with a linear kernel and an epsilon of 0.1, so these values are constant in the results reported in the table. The gamma was also set at auto. 

|            	| PCA<br>Nb of components 	| SVR<br>C 	| Weighted Final Score<br>over the 5 folds 	|
|------------	|:-----------------------:	|:--------:	|:----------------------------------------:	|
| 1004     	|          450               	|     0.002     	|     1.1469489658686098                                     	|
| 1007     	|           100              	|     0.002     	|         0.09115239389591206 |
| 1019 	|                 400        	        |       0.2   	|                 0.686931370820251	|
| 1023     	|           300              	|     0.2     	|         0.8462093717280431 |
| 1034     	|           100              	|     20.0     	|         0.7961188257851409 |
| 1038     	|           450              	|     0.002     	|         0.3530848340426855 |
| 1039     	|           450              	|     0.2     	|         0.3826339325882311 |
| 1043     	|           300              	|     0.2     	|         0.5525085362997469 |
| 1044     	|           50              	|     0.002     	|         0.09694768640213237 |
| 1048     	|           650              	|     0.2     	|         0.4505302952804157 |
| 1049     	|           250              	|     0.2     	|         0.4001809543831368 |

*Table ?: Best hyperparameters for each subject found on dysk_orig_auto60* 

For REAL-PD, we performed a gridsearch and applied global normalization. Unlike CIS-PD, there was no per-patient tuning. We tried to tuned the best parameters for each task and each subtype (phone_accelerometer, watch_accelerometer, watch_gyroscope), but the best results overall were obtained with tremor and watch_gyroscope. So we used those parameters across the three tasks and subtypes hoping for better generalization. 

| Hyperparameters  	| Values           	|
|------------------	|------------------	|
| objective        	| reg:squarederror 	|
| max_depth        	| 3                	|
| silent        	| False                	|
| learning_rate    	| 0.3             	|
| subsample        	| 0.7              	|
| colsample_bylevel 	| 0.8              	|
| colsample_bytree 	| 0.4              	|
| min_child_weight 	| 5.0                	|
| gamma              	| 1.0               	|
| reg_lambda     	| 50.0               	|
| n_estimators     	| 100               	|

*Table ?: Hyperparameters used for REAL-PD in the 4th submission, for all tasks and all subtasks.*


### Final Submission 

For the final submission, we submitted:
- `ON/OFF`:
    - CIS-PD: 4th submission
    - REAL-PD: 4th submission
- `Tremor`:
    - CIS-PD: 3rd submission
    - REAL-PD: 4th submission (because it has gridsearch)
- `Dyskinesia`: 4th submission
    - CIS-PD: 4th submission
    - REAL-PD: 4th submission

# Conclusion/Discussion

```
(suggested limit 200 words)

This section should include a short summary and any insights gained during the algorithm.
For example, which dataset was most informative? You can include future directions.
You may also add some discussion on the general performance of your methodology (if you wish) and if there were pitfalls, what are they?
```

Future work : 
- Try FAST AI to extract time series features : https://ai-fast-track.github.io/timeseries/ 
- Fine tuning on REAL-PD data directly? 
- Balacing the kfolds well? 
- gradient boosting trees implementation which is both a drop-in replacement and competitive to xgboost (sometimes slightly better) https://catboost.ai/docs/concepts/python-quickstart.html 

# References 

1. Christ, M., Braun, N., Neuffer, J., & Kempa-Liehr, A. W. (2018). Time series feature extraction on basis of scalable hypothesis tests (tsfresh–a python package). Neurocomputing, 307, 72-77. 

2. Wang, J., Song, X., & Farahi, A. Parkinson’s Disease Digital Biomarker DREAM Challenge. [Paper](https://www.biorxiv.org/content/10.1101/2020.01.13.904722v2.full) - [Challenge website](http://dreamchallenges.org/project/parkinsons-disease-digital-biomarker-dream-challenge/) - [Poster](http://midwestbigdatahub.org/wp-content/uploads/2017/09/Parkinson-DREAM-challenge-Poster.pdf).

# Authors Statement

`Please list all author’s contributions`

- Marie-Philippe Gill: Data preparation, SVR experiments, write-up, GitHub 
- Nanxin Chen: tsfresh 
- Saurabhchand Bhati: AutoEncoder, LSTM experiments 
- Sonal Joshi: activity detection
- Laureano Moro-Velazquez: MFCC, ivectors, supervision 


# Removed info 

| Hyperparameters  	| Values           	|
|------------------	|------------------	|
| objective        	| reg:squarederror 	|
| silent       	        | False               	|
| max_depth        	| 2                	|
| learning_rate    	| 0.3             	|
| subsample        	| 1.0              	|
| colsample_bytree 	| 0.8              	|
| colsample_bylevel     | 0.5                	|
| min_child_weight 	| 0.5                	|
| gamma 	        | 1.0                	|
| reg_lambda 	        | 100.0                	|
| n_estimators 	        | 100                	|

*Table ?: Best hyperparameters found with CIS-PD for submission 3.*

<table>
  <tr>
    <th></th>
    <th><span style="font-weight:bold">Submission</span></th>
    <th>Train Final Score</th>
    <th><span style="font-weight:bold">Test Final Score</span></th>
  </tr>
  <tr>
    <td rowspan="4">On/Off</td>
    <td>1</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>3</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>4</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="4">Tremor</td>
    <td>1</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>3</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>4</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="4">Dyskinesia</td>
    <td>1</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>3</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>4</td>
    <td></td>
    <td></td>
  </tr>
</table>

