import matplotlib.pyplot as plt

def visualize_misclassified_images(df, X_train):
    misclassified = df[df["Correct"] == 0]
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 15),
                             subplot_kw={'xticks': [], 'yticks': []})
    
    for i, ax in enumerate(axes.flat):
        if i >= len(misclassified):
            break
        image_path = misclassified.iloc[i]["Image"]
        image = Image.open(image_path)
        true_label = misclassified.iloc[i]["True Label"]
        pred_label = misclassified.iloc[i]["Predicted Label"]
        
        ax.imshow(image, cmap='gray')
        ax.set_title(f"True: {true_label}, Pred: {pred_label}")
    
    plt.tight_layout()
    plt.show()

if __name__ = '__main__':
    results_df = pd.read_csv('results_table.csv')
    all_images = [X_train[idx] for idx in results_df['Image Index']]
    titles = [f"True: {row['True Label']}, Pred: {row['Predicted Label']}" for _, row in results_df.iterrows()]
    accumulated_accuracy = np.mean(results_df['Correct'].cumsum().values / (results_df.index + 1))
    data = [f"Accumulated Accuracy: {accumulated_accuracy:.2%}" for _ in results_df.iterrows()]

    visualize_with_data(all_images, titles, data)