# Dataset Statistics

## Statistics of WBC
|||   WBC_100   || WBC_50 |      | WBC_10 |      | WBC_1 |      |
|:-----------:|:-----:|:----:|:----------:|:------:|:----:|:------:|:----:|:-----:|:----:|
|     | Train |      | Validation |  Train |      |  Train |      | Train |      |
|     Class    |  data | mask |    data    |  data  | mask |  data  | mask |  data | mask |
|   Basophil  |  176  |  17  |     36     |   88   |   8  |   17   |   1  |   1   |   0  |
| Eosinophils |  618  |  61  |     126    |   309  |  30  |   61   |   6  |   6   |   0  |
|  Lymphocyte |  2015 |  201 |     412    |  1007  |  100 |   201  |  20  |   20  |   2  |
|   Monocyte  |  466  |  46  |     95     |   233  |  23  |   46   |   4  |   4   |   0  |
|  Neutrophil |  5172 |  517 |    1059    |  2586  |  258 |   517  |  51  |   51  |   5  |
|    Total#   |  8447 |  842 |    1728    |  4223  |  419 |   842  |  82  |   82  |   7  |



## Statistics of CAM16 and pRCC
|  ||CAM16                             ||| pRCC  |
|:------:|:-----:|:----:|:----:|:----:|:-------:|
|        | Train       || Validation  | Test | Train |
|        | data  | mask | data | data | data  |
| Normal | 379   | 37   | 54   | 108  | -  |
| Tumor  | 378   | 37   | 54   | 108  | -     |
| Total# | 757   | 74   | 108  | 216  | 1419  |




We summarize the statistics of datasets WBC, CAM16 and pRCC in the Tables above.
In WBC_100, we provide a ratio of 5:1 data for training and validation set in each cell type. And there are three segregations (i.e., WBC_50, WBC_10, WBC_10) for WBC_100, which contain 50%, 10% and 1% data of the whole set, respectively. Both WBC and CAM16 have additional mask annotation, where 10% of the samples have masks, each name corresponding to the image name. Note, pRCC and CAM16 are offered as the pre-training set, and pRCC comes without any label.