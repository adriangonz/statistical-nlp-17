#not sure about using confusion matrix, need to be changed
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn

Y_pred = model.predict(X_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)
 
for ix in range(3):
    print(ix, confusion_matrix(np.argmax(Y_test, axis=1), y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(Y_test, axis=1), y_pred)
print(cm)
 
 
df_cm = pd.DataFrame(cm, range(3), range(3))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=False)
sn.set_context("poster")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig('Plots/confusionMatrix.png')
plt.show()