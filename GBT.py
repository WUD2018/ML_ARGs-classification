import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from numpy import savetxt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def readData(filename):

    "read data from resistome"

    #'resistome.type.rf.data.txt'
    data = pd.read_csv(filename, sep ='\\\t')
    # data = data.drop(['SampleID'],axis=1)
    grp = pd.unique(data['EnvSeason'])
    X = data[data.columns[2:]]
    label = data[data.columns[1]]

    return grp,X, label

def clf_model():
    # {'criterion': 'friedman_mse', 'learning_rate': 2, 'max_depth': 5,
    # 'max_features': 'log2', 'min_samples_split': 2,
    #  'n_estimators': 10}

    clf = GradientBoostingClassifier(n_estimators=120, learning_rate=0.01,
                                     max_depth = 6, random_state = 123,
                                     criterion='friedman_mse', max_features='log2',
                                     min_samples_split=5)

    return clf

def main():
    grp, X, label = readData('ML.table.txt')
    clf = clf_model()

    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42)

    Mtrcs = []
    # set each env as positive label in turn
    Mtrcs_t = []
    colornames = ["red", "blue", "yellow", "green"]
    for (g, colorname) in zip(grp, colornames):

        y = np.zeros(y_train.shape)
        y[y_train != g] = 0
        y[y_train == g] = 1
        y = np.array(y, dtype=int)
        y_t = np.zeros(y_test.shape)
        y_t[y_test != g] = 0
        y_t[y_test == g] = 1
        y_t = np.array(y_t, dtype=int)

        # cross validation
        cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
        Pred = []
        Pred_p = []
        Real = []
        Mtrcs_each_g = []

        Test_Pred = []
        Test_Pred_p = []
        Test_Real = []
        Test_Mtrcs_each_g = []

        for (Train, Valid), i in zip(cv.split(X_train, y), range(5)):
            clf.fit(X_train.iloc[Train], y[Train])
            y_pred = clf.predict(X_train.iloc[Valid])
            y_pred_proba = clf.predict_proba(X_train.iloc[Valid])
            mtrcs = precision_recall_fscore_support(y[Valid], y_pred, pos_label=1, average='macro')
            acc = accuracy_score(y[Valid], y_pred)
            Mtrcs_each_g.append([acc] + list(mtrcs[:-1]))
            Pred = Pred + y_pred.tolist()
            Pred_p = Pred_p + y_pred_proba.tolist()
            Real = Real + y[Valid].tolist()

            Test_y_pred = clf.predict(X_test)
            Test_y_pred_proba = clf.predict_proba(X_test)
            mtrcs_t = precision_recall_fscore_support(y_t, Test_y_pred, pos_label=1, average='macro')
            acc_t = accuracy_score(y_t, Test_y_pred)
            Test_Mtrcs_each_g.append([acc_t] + list(mtrcs_t[:-1]))
            Test_Pred = Test_Pred + Test_y_pred.tolist()
            Test_Pred_p = Test_Pred_p + Test_y_pred_proba.tolist()
            Test_Real = Test_Real + y_t.tolist()

        Mtrcs.append(Mtrcs_each_g)
        # Pred = np.asarray(Pred)
        Real = np.asarray(Real)
        Pred_p = np.asarray(Pred_p)
        cm = confusion_matrix(Real, Pred)
        print("{}:".format(g), cm)

        Mtrcs_t.append(Test_Mtrcs_each_g)
        Test_Real = np.asarray(Test_Real)
        Test_Pred_p = np.asarray(Test_Pred_p)

        # #ROC plot for each env
        # metrics.RocCurveDisplay.from_predictions(
        #     Real,
        #     Pred_p[:,1],
        #     name=f"{g} vs the rest1",
        #     color="darkorange",
        # )
        #
        # plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        # plt.axis("square")
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.title(f"One-vs-Rest ROC curves:\n{g} vs (Other groups)")
        # plt.legend()
        # # plt.savefig(f"RF_ROC_figure/12/{g}_ROC_12.png",dpi=600)
        # plt.show()

        # Plot the confusion matrix.
        # sns.heatmap(cm,
        #             annot=True,
        #             fmt='g',
        #             xticklabels=['Not {}'.format(g), '{}'.format(g)],
        #             yticklabels=['Not {}'.format(g), '{}'.format(g)])
        # plt.ylabel('Predicted Label', fontsize=13)
        # plt.xlabel('Actual Label', fontsize=13)
        # plt.title('Confusion Matrix of {}'.format(g), fontsize=17)
        # plt.savefig("confusion_matrix_{}_plot.pdf".format(g))
        # acc = accuracy_score(Real, Pred)
        # # plt.text(1, 1, "Accuracy: {:.2f}".format(acc), ha="center")
        # plt.show()

        ft = clf.feature_importances_
        ft_pandas = pd.DataFrame(ft, index=list(X.columns.values),columns=["feature importance"])
        ft_pandas.to_csv('./ML_Air_Swab_Sum_Win/GBT/GBT_Feature_rank_{}.csv'.format(g))

        # ROC plot for all envs
        fpr, tpr, thresholds = metrics.roc_curve(Test_Real, Test_Pred_p[:, 1], pos_label=1)

        plt.plot(fpr, tpr, lw=2, label='{}(AUC={:.3f})'.format(g, metrics.auc(fpr, tpr)),
                 color=colorname)
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis('square')
    plt.xlim([-0.01, 1.02])
    plt.ylim([-0.01, 1.02])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plt.legend(loc='lower right', fontsize=9)

    plt.savefig("./ML_Air_Swab_Sum_Win/GBT/ROC_curve2.pdf", dpi=600)
    plt.show()

    # # feature importance for each env
    # ft = clf.feature_importances_
    # ft_pandas = pd.DataFrame(ft, index=list(X.columns.values),columns=["feature importance"])
    # ft_pandas.to_csv(f'RF_Feature importance/12/{g}_feature_rank_12.csv')

    # save envaluation metrics for each env
    # Mtrcs = np.asarray(Mtrcs)
    # final_score = np.mean(Mtrcs,1)
    # # final_score_std = np.std(Mtrcs,1)
    # panda_Mtrcs = pd.DataFrame(data = final_score, index=grp.tolist(),
    #                         columns = ["Accuracy","Precision","Recall", "F1score"])
    # panda_Mtrcs.to_csv('./Air_Swab_Sum_Win/RF_Precision_Recall_F1.csv')

    Mtrcs_t = np.asarray(Mtrcs_t)
    final_score_test = np.mean(Mtrcs_t, 1)
    # final_score_std = np.std(Mtrcs,1)
    panda_Mtrcs_test = pd.DataFrame(data=final_score_test, index=grp.tolist(),
                                    columns=["Accuracy", "Precision", "Recall", "F1score"])
    panda_Mtrcs_test.to_csv('./ML_Air_Swab_Sum_Win/GBT/GBT_Precision_Recall_F1_test2.csv')


if __name__ == "__main__":
    main()
    print('end')