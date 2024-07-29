1. I combining 2 loss fuction to improve performance: Cross-Entropy Loss and Dice Loss.
- Cross-Entropy Loss: 
    - Pos: widely used for classification problems. It measures the diffirient between model's prediction and ground_truth
    - Cons: may not perform well on imbalanced data because it does not directly consider the class ratios.
- Dice Loss:
    - Pos: Specifically designed for image segmentation tasks, optimizing the overlap between predicted regions and actual labels. It's very effective for imbalanced data
    - Cos: May not provide strong gradient signals for all pixels, especially in the early stages of training