
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, multivariate_normal

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual,IntSlider

data = np.loadtxt('wine.data.txt', delimiter=',')

featurenames = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash','Magnesium', 'Total phenols', 
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 
                'OD280/OD315 of diluted wines', 'Proline']

np.random.seed(0)
perm = np.random.permutation(178)
trainx = data[perm[0:130],1:14]
trainy = data[perm[0:130],0]
testx = data[perm[130:178], 1:14]
testy = data[perm[130:178],0]
sum(trainy==1), sum(trainy==2), sum(trainy==3)



    


    
# # question 1
# print(sum(testy==1), sum(testy==2), sum(testy==3))


# # question 2
# @interact_manual( feature=IntSlider(0,0,12), label=IntSlider(1,1,3))
# def density_plot(feature, label):
#     plt.hist(trainx[trainy==label,feature],normed=True)
#     #
#     mu = np.mean(trainx[trainy==label,feature]) # mean
#     var = np.var(trainx[trainy==label,feature]) # variance
#     std = np.sqrt(var) # standard deviation
#     #
#     x_axis = np.linspace(mu - 3*std, mu + 3*std, 1000)
#     plt.plot(x_axis, norm.pdf(x_axis,mu,std), 'r', lw=2)
#     plt.title("Winery "+str(label) )
#     plt.xlabel(featurenames[feature], fontsize=14, color='red')
#     plt.ylabel('Density', fontsize=14, color='red')
#     plt.show()
    
# # modify this cell
# std = np.zeros(13)
# for feature in range(0,13):
#     std[feature] = np.std(trainx[trainy==1,feature])
# std

# import numpy as np

# # Filter training data for winery 1
# winery1_data = trainx[trainy == 1]  # Select only rows where trainy == 1

# # Compute standard deviations for each feature (column)
# std_devs = np.std(winery1_data, axis=0)

# # Find the feature index with the smallest standard deviation
# smallest_std_feature = np.argmin(std_devs)

# print("Feature with the smallest standard deviation for winery 1:", smallest_std_feature)



# #question 3
# # Assumes y takes on values 1,2,3
# def fit_generative_model(x,y,feature):
#     k = 3 # number of classes
#     mu = np.zeros(k+1) # list of means
#     var = np.zeros(k+1) # list of variances
#     pi = np.zeros(k+1) # list of class weights
#     for label in range(1,k+1):
#         indices = (y==label)
#         mu[label] = np.mean(x[indices,feature])
#         var[label] = np.var(x[indices,feature])
#         pi[label] = float(sum(indices))/float(len(y))
#     return mu, var, pi

# feature = 0 # 'alcohol'
# mu, var, pi = fit_generative_model(trainx, trainy, feature)
# print (pi[1:])

# @interact_manual( feature=IntSlider(0,0,12) )
# def show_densities(feature):
#     mu, var, pi = fit_generative_model(trainx, trainy, feature)
#     colors = ['r', 'k', 'g']
#     for label in range(1,4):
#         m = mu[label]
#         s = np.sqrt(var[label])
#         x_axis = np.linspace(m - 3*s, m+3*s, 1000)
#         plt.plot(x_axis, norm.pdf(x_axis,m,s), colors[label-1], label="class " + str(label))
#     plt.xlabel(featurenames[feature], fontsize=14, color='red')
#     plt.ylabel('Density', fontsize=14, color='red')
#     plt.legend()
#     plt.show()
    
# # Function to compute histogram overlap between two distributions
# def compute_overlap(mu, var, label1, label2):
#     std1, std2 = np.sqrt(var[label1]), np.sqrt(var[label2])
#     mean1, mean2 = mu[label1], mu[label2]
    
#     # Define a common x range
#     x_axis = np.linspace(min(mean1 - 3*std1, mean2 - 3*std2), 
#                          max(mean1 + 3*std1, mean2 + 3*std2), 1000)
    
#     # Compute density functions
#     pdf1 = norm.pdf(x_axis, mean1, std1)
#     pdf2 = norm.pdf(x_axis, mean2, std2)
    
#     # Compute intersection
#     overlap = np.trapz(np.minimum(pdf1, pdf2), x_axis)
#     return overlap

# # Find answers to the three questions
# overlaps = []
# spreads = []
# separations = []

# for feature in range(13):
#     mu, var, pi = fit_generative_model(trainx, trainy, feature)
    
#     # 1️⃣ Compute overlap between classes 1 and 3
#     overlap_13 = compute_overlap(mu, var, 1, 3)
#     overlaps.append(overlap_13)
    
#     # 2️⃣ Compute standard deviation of class 3
#     spreads.append(np.sqrt(var[3]))
    
#     # 3️⃣ Compute class separation (sum of pairwise mean differences)
#     separation = (abs(mu[1] - mu[2]) + abs(mu[1] - mu[3]) + abs(mu[2] - mu[3])) / 3
#     separations.append(separation)

# # Get feature indices for each answer
# feature_overlap = np.argmax(overlaps)  # Most overlap between class 1 and 3
# feature_spread = np.argmax(spreads)  # Most spread out class 3
# feature_separation = np.argmax(separations)  # Best separation

# # Print final answers
# print(f"Feature with most overlap between classes 1 and 3: {feature_overlap} ({featurenames[feature_overlap]})")
# print(f"Feature where class 3 is most spread out: {feature_spread} ({featurenames[feature_spread]})")
# print(f"Feature where classes are best separated: {feature_separation} ({featurenames[feature_separation]})")




#question4

@interact( feature=IntSlider(0,0,12) )
def test_model(feature):
    mu, var, pi = fit_generative_model(trainx, trainy, feature)

    k = 3 # Labels 1,2,...,k
    n_test = len(testy) # Number of test points
    score = np.zeros((n_test,k+1))
    for i in range(0,n_test):
        for label in range(1,k+1):
            score[i,label] = np.log(pi[label]) + \
            norm.logpdf(testx[i,feature], mu[label], np.sqrt(var[label]))
    predictions = np.argmax(score[:,1:4], axis=1) + 1
    # Finally, tally up score
    errors = np.sum(predictions != testy)
    print("Test error using feature " + featurenames[feature] + ": " + str(errors) + "/" + str(n_test))
    
    
def fit_generative_model(x, y, feature):
    k = 3  # number of classes
    mu = np.zeros(k+1)  # list of means
    var = np.zeros(k+1)  # list of variances
    pi = np.zeros(k+1)  # list of class weights
    for label in range(1, k+1):
        indices = (y == label)
        mu[label] = np.mean(x[indices, feature])
        var[label] = np.var(x[indices, feature])
        pi[label] = float(sum(indices)) / float(len(y))
    return mu, var, pi

    
# Compute training and test errors for each feature

train_errors = []
test_errors = []

for feature in range(13):  # Loop through all features
    mu, var, pi = fit_generative_model(trainx, trainy, feature)

    k = 3  # Labels 1,2,3
    n_train = len(trainy)
    n_test = len(testy)

    # Compute training error
    train_score = np.zeros((n_train, k+1))
    for i in range(n_train):
        for label in range(1, k+1):
            train_score[i, label] = np.log(pi[label]) + norm.logpdf(
                trainx[i, feature], mu[label], np.sqrt(var[label])
            )
    train_predictions = np.argmax(train_score[:, 1:4], axis=1) + 1
    train_error = np.sum(train_predictions != trainy)
    train_errors.append(train_error)

    # Compute test error
    test_score = np.zeros((n_test, k+1))
    for i in range(n_test):
        for label in range(1, k+1):
            test_score[i, label] = np.log(pi[label]) + norm.logpdf(
                testx[i, feature], mu[label], np.sqrt(var[label])
            )
    test_predictions = np.argmax(test_score[:, 1:4], axis=1) + 1
    test_error = np.sum(test_predictions != testy)
    test_errors.append(test_error)

# Display results
print("Feature-wise Training and Test Errors:")
for i in range(13):
    print(f"{featurenames[i]} - Training Error: {train_errors[i]}/{n_train}, Test Error: {test_errors[i]}/{n_test}")

# Plot the training and test errors
plt.figure(figsize=(10, 5))
plt.plot(range(13), train_errors, marker='o', label="Training Error", color="blue")
plt.plot(range(13), test_errors, marker='s', label="Test Error", color="red")
plt.xticks(range(13), featurenames, rotation=45, ha="right")
plt.xlabel("Feature")
plt.ylabel("Number of Errors")
plt.legend()
plt.title("Training and Test Errors for Each Feature")
plt.show()
    