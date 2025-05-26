# Processed Data Dictionary

The purpose of this file is to provide a clear and detailed description of the columns in the processed dataset Here you will find definitions and explanations of the identified fields to facilitate understanding and analysis of the data, ensuring that all users have a consistent and coordinated view of the information contained.

- General features.

| Name                       | Type    | Description                                                                  |
| -------------------------- | ------- | ---------------------------------------------------------------------------- |
| video_published_at         | object  | The date and time when the video was published                               |
| days_publishewd            | int64   | Number of days the video has been published                                  |
| video_duration             | object  | Duration of the video in seconds                                             |
| published_dayofweek        | int64   | Day of the week when the video was published (Monda =0,...,Sunday=6)         |
| published_hour             | int64   | Hour of the day when the video was published (0 to 23)                       |
| days_to_trend              | int64   | Total number of days until the video becomes a trend                         |
| video_view_count           | float64 | Total number of views for the video                                          |
| video_like_count           | float64 | Total number of likes for the video                                          |
| video_comment_count        | float64 | Total number of comments on the video                                        |
| channel_view_count         | float64 | Total number of views across all videos on the channel                       |
| channel_subscriber_count   | float64 | Total number of subscribers to the channel                                   |
| video_title_length         | int64   | Number of characters in the video title                                      |
| video_tag_count            | int64   | Number of tags associated with the video                                     |

Each feature below is a PCA component representing video category embeddings Some of these features may not appear in the dataset depending on the number of components detected (Defaul maximum amount of components to be consider: 20).

| Name                       | Type    | Description                                                                  |
| -------------------------- | ------- | ---------------------------------------------------------------------------- |
| video_category_pca_0       | float64 | First principal component capturing the most variance in category embeddings |
| video_category_pca_1       | float64 | Second principal component from the video category embedding                 |
| video_category_pca_2       | float64 | Third principal component reflecting latent patterns in video categories     |
| video_category_pca_3       | float64 | Fourth component capturing semantic variation in category type               |
| video_category_pca_4       | float64 | Fifth component describing complex combinations of category semantics        |
| video_category_pca_5       | float64 | Sixth principal component from category embeddings                           |
| video_category_pca_6       | float64 | Seventh PCA feature summarizing category relationships                       |
| video_category_pca_7       | float64 | Eighth PCA feature from video category vectors                               |
| video_category_pca_8       | float64 | Ninth component capturing category-related variance                          |
| video_category_pca_9       | float64 | Tenth PCA feature derived from category embedding space                      |
| video_category_pca_10      | float64 | Eleventh component reflecting subtle category nuances                        |
| video_category_pca_11      | float64 | Twelfth PCA dimension from video category embedding                          |
| video_category_pca_12      | float64 | Thirteenth component encoding abstract category traits                       |
| video_category_pca_13      | float64 | Fourteenth PCA projection of video category features                         |
| video_category_pca_14      | float64 | Fifteenth component summarizing category embedding interactions              |
| video_category_pca_15      | float64 | Sixteenth PCA dimension from video category embeddings                       |
| video_category_pca_16      | float64 | Seventeenth latent feature from PCA of category embeddings                   |
| video_category_pca_17      | float64 | Eighteenth component capturing complex semantic patterns                     |
| video_category_pca_18      | float64 | Nineteenth principal component of category representations                   |
| video_category_pca_19      | float64 | Twentieth component capturing residual semantic variance in categories       |

- If --translate=True is specified.

| Name                       | Type    | Description                                                                  |
| -------------------------- | ------- | ---------------------------------------------------------------------------- |
| video_title_language       | object  | Language of the video title as displayed on YouTube                          |
| video_title_translated     | object  | Translated title of the video                                                |

- If --vectorize=True is specified.

Each feature below is a binary (int64) flag indicating whether terms obtyained from applying TfidfVectorizer was detected in the video title (0 = Not detected, 1 = Detected). Some of these features may not appear in the dataset (Defaul maximum amount of features to be consider by the vectorizer: 100).

| Name                       | Type    | Description                                                                  |
| -------------------------- | ------- | ---------------------------------------------------------------------------- |
| term_0                     | int64   | Indicator the presence of keyword term 0 in video title                      |
| term_1                     | int64   | Indicator the presence of keyword term 1 in video title                      |
| term_2                     | int64   | Indicator the presence of keyword term 2 in video title                      |
| term_3                     | int64   | Indicator the presence of keyword term 3 in video title                      |
| term_4                     | int64   | Indicator the presence of keyword term 4 in video title                      |
| term_5                     | int64   | Indicator the presence of keyword term 5 in video title                      |
| term_6                     | int64   | Indicator the presence of keyword term 6 in video title                      |
| term_7                     | int64   | Indicator the presence of keyword term 7 in video title                      |
| term_8                     | int64   | Indicator the presence of keyword term 8 in video title                      |
| term_9                     | int64   | Indicator the presence of keyword term 9 in video title                      |
| term_10                    | int64   | Indicator the presence of keyword term 10 in video title                     |
| term_11                    | int64   | Indicator the presence of keyword term 11 in video title                     |
| term_12                    | int64   | Indicator the presence of keyword term 12 in video title                     |
| term_13                    | int64   | Indicator the presence of keyword term 13 in video title                     |
| term_14                    | int64   | Indicator the presence of keyword term 14 in video title                     |
| term_15                    | int64   | Indicator the presence of keyword term 15 in video title                     |
| term_16                    | int64   | Indicator the presence of keyword term 16 in video title                     |
| term_17                    | int64   | Indicator the presence of keyword term 17 in video title                     |
| term_18                    | int64   | Indicator the presence of keyword term 18 in video title                     |
| term_19                    | int64   | Indicator the presence of keyword term 19 in video title                     |
| term_20                    | int64   | Indicator the presence of keyword term 20 in video title                     |
| term_21                    | int64   | Indicator the presence of keyword term 21 in video title                     |
| term_22                    | int64   | Indicator the presence of keyword term 22 in video title                     |
| term_23                    | int64   | Indicator the presence of keyword term 23 in video title                     |
| term_24                    | int64   | Indicator the presence of keyword term 24 in video title                     |
| term_25                    | int64   | Indicator the presence of keyword term 25 in video title                     |
| term_26                    | int64   | Indicator the presence of keyword term 26 in video title                     |
| term_27                    | int64   | Indicator the presence of keyword term 27 in video title                     |
| term_28                    | int64   | Indicator the presence of keyword term 28 in video title                     |
| term_29                    | int64   | Indicator the presence of keyword term 29 in video title                     |
| term_30                    | int64   | Indicator the presence of keyword term 30 in video title                     |
| term_31                    | int64   | Indicator the presence of keyword term 31 in video title                     |
| term_32                    | int64   | Indicator the presence of keyword term 32 in video title                     |
| term_33                    | int64   | Indicator the presence of keyword term 33 in video title                     |
| term_34                    | int64   | Indicator the presence of keyword term 34 in video title                     |
| term_35                    | int64   | Indicator the presence of keyword term 35 in video title                     |
| term_36                    | int64   | Indicator the presence of keyword term 36 in video title                     |
| term_37                    | int64   | Indicator the presence of keyword term 37 in video title                     |
| term_38                    | int64   | Indicator the presence of keyword term 38 in video title                     |
| term_39                    | int64   | Indicator the presence of keyword term 39 in video title                     |
| term_40                    | int64   | Indicator the presence of keyword term 40 in video title                     |
| term_41                    | int64   | Indicator the presence of keyword term 41 in video title                     |
| term_42                    | int64   | Indicator the presence of keyword term 42 in video title                     |
| term_43                    | int64   | Indicator the presence of keyword term 43 in video title                     |
| term_44                    | int64   | Indicator the presence of keyword term 44 in video title                     |
| term_45                    | int64   | Indicator the presence of keyword term 45 in video title                     |
| term_46                    | int64   | Indicator the presence of keyword term 46 in video title                     |
| term_47                    | int64   | Indicator the presence of keyword term 47 in video title                     |
| term_48                    | int64   | Indicator the presence of keyword term 48 in video title                     |
| term_49                    | int64   | Indicator the presence of keyword term 49 in video title                     |
| term_50                    | int64   | Indicator the presence of keyword term 50 in video title                     |
| term_51                    | int64   | Indicator the presence of keyword term 51 in video title                     |
| term_52                    | int64   | Indicator the presence of keyword term 52 in video title                     |
| term_53                    | int64   | Indicator the presence of keyword term 53 in video title                     |
| term_54                    | int64   | Indicator the presence of keyword term 54 in video title                     |
| term_55                    | int64   | Indicator the presence of keyword term 55 in video title                     |
| term_56                    | int64   | Indicator the presence of keyword term 56 in video title                     |
| term_57                    | int64   | Indicator the presence of keyword term 57 in video title                     |
| term_58                    | int64   | Indicator the presence of keyword term 58 in video title                     |
| term_59                    | int64   | Indicator the presence of keyword term 59 in video title                     |
| term_60                    | int64   | Indicator the presence of keyword term 60 in video title                     |
| term_61                    | int64   | Indicator the presence of keyword term 61 in video title                     |
| term_62                    | int64   | Indicator the presence of keyword term 62 in video title                     |
| term_63                    | int64   | Indicator the presence of keyword term 63 in video title                     |
| term_64                    | int64   | Indicator the presence of keyword term 64 in video title                     |
| term_65                    | int64   | Indicator the presence of keyword term 65 in video title                     |
| term_66                    | int64   | Indicator the presence of keyword term 66 in video title                     |
| term_67                    | int64   | Indicator the presence of keyword term 67 in video title                     |
| term_68                    | int64   | Indicator the presence of keyword term 68 in video title                     |
| term_69                    | int64   | Indicator the presence of keyword term 69 in video title                     |
| term_70                    | int64   | Indicator the presence of keyword term 70 in video title                     |
| term_71                    | int64   | Indicator the presence of keyword term 71 in video title                     |
| term_72                    | int64   | Indicator the presence of keyword term 72 in video title                     |
| term_73                    | int64   | Indicator the presence of keyword term 73 in video title                     |
| term_74                    | int64   | Indicator the presence of keyword term 74 in video title                     |
| term_75                    | int64   | Indicator the presence of keyword term 75 in video title                     |
| term_76                    | int64   | Indicator the presence of keyword term 76 in video title                     |
| term_77                    | int64   | Indicator the presence of keyword term 77 in video title                     |
| term_78                    | int64   | Indicator the presence of keyword term 78 in video title                     |
| term_79                    | int64   | Indicator the presence of keyword term 79 in video title                     |
| term_80                    | int64   | Indicator the presence of keyword term 80 in video title                     |
| term_81                    | int64   | Indicator the presence of keyword term 81 in video title                     |
| term_82                    | int64   | Indicator the presence of keyword term 82 in video title                     |
| term_83                    | int64   | Indicator the presence of keyword term 83 in video title                     |
| term_84                    | int64   | Indicator the presence of keyword term 84 in video title                     |
| term_85                    | int64   | Indicator the presence of keyword term 85 in video title                     |
| term_86                    | int64   | Indicator the presence of keyword term 86 in video title                     |
| term_87                    | int64   | Indicator the presence of keyword term 87 in video title                     |
| term_88                    | int64   | Indicator the presence of keyword term 88 in video title                     |
| term_89                    | int64   | Indicator the presence of keyword term 89 in video title                     |
| term_90                    | int64   | Indicator the presence of keyword term 90 in video title                     |
| term_91                    | int64   | Indicator the presence of keyword term 91 in video title                     |
| term_92                    | int64   | Indicator the presence of keyword term 92 in video title                     |
| term_93                    | int64   | Indicator the presence of keyword term 93 in video title                     |
| term_94                    | int64   | Indicator the presence of keyword term 94 in video title                     |
| term_95                    | int64   | Indicator the presence of keyword term 95 in video title                     |
| term_96                    | int64   | Indicator the presence of keyword term 96 in video title                     |
| term_97                    | int64   | Indicator the presence of keyword term 97 in video title                     |
| term_98                    | int64   | Indicator the presence of keyword term 98 in video title                     |
| term_99                    | int64   | Indicator the presence of keyword term 99 in video title                     |



- If --analyze=True is specified.

| Name                       | Type    | Description                                                                  |
| -------------------------- | ------- | ---------------------------------------------------------------------------- |
| sentiment_score            | float64 | Overall sentiment score from a sentiment analysis model                      |
| sentiment_negative         | float64 | Probability or score of negative sentiment                                   |
| sentiment_neutral          | float64 | Probability or score of neutral sentiment                                    |
| sentiment_positive         | float64 | Probability or score of positive sentiment                                   |

- If --stats=True is specified.

| Name                       | Type    | Description                                                                  |
| -------------------------- | ------- | ---------------------------------------------------------------------------- |
| thumbnail_brightness       | float64 | Average brightness of the video thumbnail                                    |
| thumbnail_contrast         | float64 | Contrast level in the thumbnail                                              |
| thumbnail_saturation       | float64 | Color saturation of the thumbnail                                            |

- If --detect=True is specified.

Each feature below is a binary (int64) flag indicating whether at least one object of the specific class was detected in the thumbnail (0 = Not detected, 1 = Detected). Some of these features may not appear in the dataset if a minimum number of objects of the corresponding class were not detected in the thumbnail set.

| Name                       | Type    | Description                                                                  |
| -------------------------- | ------- | ---------------------------------------------------------------------------- |
| thumbnail_person           | int64   | Indicator if an object of the class person was detected in thumbnail         |
| thumbnail_bicycle          | int64   | Indicator if an object of the class bicycle was detected in thumbnail        |
| thumbnail_car              | int64   | Indicator if an object of the class car was detected in thumbnail            |
| thumbnail_motorcycle       | int64   | Indicator if an object of the class motorcycle was detected                  |
| thumbnail_airplane         | int64   | Indicator if an object of the class airplane was detected                    |
| thumbnail_bus              | int64   | Indicator if an object of the class bus was detected                         |
| thumbnail_train            | int64   | Indicator if an object of the class train was detected                       |
| thumbnail_truck            | int64   | Indicator if an object of the class truck was detected                       |
| thumbnail_boat             | int64   | Indicator if an object of the class boat was detected                        |
| thumbnail_traffic_light    | int64   | Indicator if an object of the class traffic light was detected               |
| thumbnail_fire_hydrant     | int64   | Indicator if an object of the class fire hydrant was detected                |
| thumbnail_stop_sign        | int64   | Indicator if an object of the class stop sign was detected                   |
| thumbnail_parking_meter    | int64   | Indicator if an object of the class parking meter was detected               |
| thumbnail_bench            | int64   | Indicator if an object of the class bench was detected                       |
| thumbnail_bird             | int64   | Indicator if an object of the class bird was detected                        |
| thumbnail_cat              | int64   | Indicator if an object of the class cat was detected                         |
| thumbnail_dog              | int64   | Indicator if an object of the class dog was detected                         |
| thumbnail_horse            | int64   | Indicator if an object of the class horse was detected                       |
| thumbnail_sheep            | int64   | Indicator if an object of the class sheep was detected                       |
| thumbnail_cow              | int64   | Indicator if an object of the class cow was detected                         |
| thumbnail_elephant         | int64   | Indicator if an object of the class elephant was detected                    |
| thumbnail_bear             | int64   | Indicator if an object of the class bear was detected                        |
| thumbnail_zebra            | int64   | Indicator if an object of the class zebra was detected                       |
| thumbnail_giraffe          | int64   | Indicator if an object of the class giraffe was detected                     |
| thumbnail_backpack         | int64   | Indicator if an object of the class backpack was detected                    |
| thumbnail_umbrella         | int64   | Indicator if an object of the class umbrella was detected                    |
| thumbnail_handbag          | int64   | Indicator if an object of the class handbag was detected                     |
| thumbnail_tie              | int64   | Indicator if an object of the class tie was detected                         |
| thumbnail_suitcase         | int64   | Indicator if an object of the class suitcase was detected                    |
| thumbnail_frizbee          | int64   | Indicator if an object of the class frizbee was detected                     |
| thumbnail_skis             | int64   | Indicator if an object of the class skis was detected                        |
| thumbnail_snowboard        | int64   | Indicator if an object of the class snowboard was detected                   |
| thumbnail_sports_ball      | int64   | Indicator if an object of the class sports ball was detected                 |
| thumbnail_kite             | int64   | Indicator if an object of the class kite was detected                        |
| thumbnail_baseball_bat     | int64   | Indicator if an object of the class baseball bat was detected                |
| thumbnail_baseball_glove   | int64   | Indicator if an object of the class baseball glove was detected              |
| thumbnail_skateboard       | int64   | Indicator if an object of the class skateboard was detected                  |
| thumbnail_surfboard        | int64   | Indicator if an object of the class surfboard was detected                   |
| thumbnail_tennis_racket    | int64   | Indicator if an object of the class tennis racket was detected               |
| thumbnail_bottle           | int64   | Indicator if an object of the class bottle was detected                      |
| thumbnail_wine_glass       | int64   | Indicator if an object of the class wine glass was detected                  |
| thumbnail_cup              | int64   | Indicator if an object of the class cup was detected                         |
| thumbnail_fork             | int64   | Indicator if an object of the class fork was detected                        |
| thumbnail_knife            | int64   | Indicator if an object of the class knife was detected                       |
| thumbnail_spoon            | int64   | Indicator if an object of the class spoon was detected                       |
| thumbnail_bowl             | int64   | Indicator if an object of the class bowl was detected                        |
| thumbnail_banana           | int64   | Indicator if an object of the class banana was detected                      |
| thumbnail_apple            | int64   | Indicator if an object of the class apple was detected                       |
| thumbnail_sandwich         | int64   | Indicator if an object of the class sandwich was detected                    |
| thumbnail_orange           | int64   | Indicator if an object of the class orange was detected                      |
| thumbnail_broccoli         | int64   | Indicator if an object of the class broccoli was detected                    |
| thumbnail_carrot           | int64   | Indicator if an object of the class carrot was detected                      |
| thumbnail_hot_dog          | int64   | Indicator if an object of the class hot dog was detected                     |
| thumbnail_pizza            | int64   | Indicator if an object of the class pizza was detected                       |
| thumbnail_donut            | int64   | Indicator if an object of the class donut was detected                       |
| thumbnail_cake             | int64   | Indicator if an object of the class cake was detected                        |
| thumbnail_chair            | int64   | Indicator if an object of the class chair was detected                       |
| thumbnail_couch            | int64   | Indicator if an object of the class couch was detected                       |
| thumbnail_potted_plant     | int64   | Indicator if an object of the class potted plant was detected                |
| thumbnail_bed              | int64   | Indicator if an object of the class bed was detected                         |
| thumbnail_dining_table     | int64   | Indicator if an object of the class dining table was detected                |
| thumbnail_toilet           | int64   | Indicator if an object of the class toilet was detected                      |
| thumbnail_tv               | int64   | Indicator if an object of the class tv was detected                          |
| thumbnail_laptop           | int64   | Indicator if an object of the class laptop was detected                      |
| thumbnail_mouse            | int64   | Indicator if an object of the class mouse was detected                       |
| thumbnail_remote           | int64   | Indicator if an object of the class remote was detected                      |
| thumbnail_keyboard         | int64   | Indicator if an object of the class keyboard was detected                    |
| thumbnail_cell_phone       | int64   | Indicator if an object of the class cell phone was detected                  |
| thumbnail_microwave        | int64   | Indicator if an object of the class microwave was detected                   |
| thumbnail_oven             | int64   | Indicator if an object of the class oven was detected                        |
| thumbnail_toaster          | int64   | Indicator if an object of the class toaster was detected                     |
| thumbnail_sink             | int64   | Indicator if an object of the class sink was detected                        |
| thumbnail_refrigerator     | int64   | Indicator if an object of the class refrigerator was detected                |
| thumbnail_book             | int64   | Indicator if an object of the class book was detected                        |
| thumbnail_clock            | int64   | Indicator if an object of the class clock was detected                       |
| thumbnail_vase             | int64   | Indicator if an object of the class vase was detected                        |
| thumbnail_scissors         | int64   | Indicator if an object of the class scissors was detected                    |
| thumbnail_teddy_bear       | int64   | Indicator if an object of the class teddy bear was detected                  |
| thumbnail_hair_drier       | int64   | Indicator if an object of the class hair drier was detected                  |
| thumbnail_toothbrush       | int64   | Indicator if an object of the class toothbrush was detected                  |

- If --embed=True is specified.

Each feature below is a PCA component representing thumbnail embeddings Some of these features may not appear in the dataset depending on the number of components detected (Defaul maximum amount of components to be consider: 100).

| Name                       | Type    | Description                                                                  |
| -------------------------- | ------- | ---------------------------------------------------------------------------- |
| thumb_pca_0                | float64 | First PCA component capturing the dominant visual variance in thumbnails     |
| thumb_pca_1                | float64 | Second principal component of thumbnail pixel data                           |
| thumb_pca_2                | float64 | Third component representing major spatial or color variation                |
| thumb_pca_3                | float64 | Fourth component summarizing image structure and contrast                    |
| thumb_pca_4                | float64 | Fifth PCA component capturing high-level visual patterns                     |
| thumb_pca_5                | float64 | Sixth component from visual pixel variance in thumbnails                     |
| thumb_pca_6                | float64 | Seventh PCA feature summarizing shape or edge information                    |
| thumb_pca_7                | float64 | Eighth component reflecting brightness or saturation differences             |
| thumb_pca_8                | float64 | Ninth PCA feature highlighting color or composition traits                   |
| thumb_pca_9                | float64 | Tenth component describing key visual abstractions                           |
| thumb_pca_10               | float64 | Eleventh PCA projection capturing subtle image differences                   |
| thumb_pca_11               | float64 | Twelfth feature from PCA on thumbnail pixel patterns                         |
| thumb_pca_12               | float64 | Thirteenth component encoding contrast and texture                           |
| thumb_pca_13               | float64 | Fourteenth visual feature derived from PCA                                   |
| thumb_pca_14               | float64 | Fifteenth component summarizing regional color intensity                     |
| thumb_pca_15               | float64 | Sixteenth PCA feature describing shape orientation                           |
| thumb_pca_16               | float64 | Seventeenth PCA component of visual input                                    |
| thumb_pca_17               | float64 | Eighteenth principal direction from thumbnail features                       |
| thumb_pca_18               | float64 | Nineteenth image-based component from PCA                                    |
| thumb_pca_19               | float64 | Twentieth component capturing residual visual variance                       |
| thumb_pca_20               | float64 | Twenty-first PCA feature from image pixel structure                          |
| thumb_pca_21               | float64 | Twenty-second component from color, texture, and edge data                   |
| thumb_pca_22               | float64 | Twenty-third PCA feature of thumbnail visual composition                     |
| thumb_pca_23               | float64 | Twenty-fourth component summarizing low-level image features                 |
| thumb_pca_24               | float64 | Twenty-fifth PCA direction from thumbnail embedding space                    |
| thumb_pca_25               | float64 | Twenty-sixth component reflecting subtle contrasts                           |
| thumb_pca_26               | float64 | Twenty-seventh PCA projection from visual thumbnail data                     |
| thumb_pca_27               | float64 | Twenty-eighth image-based latent component                                   |
| thumb_pca_28               | float64 | Twenty-ninth PCA feature capturing visual diversity                          |
| thumb_pca_29               | float64 | Thirtieth component encoding global image structure                          |
| thumb_pca_30               | float64 | Thirty-first component summarizing minor visual patterns                     |
| thumb_pca_31               | float64 | Thirty-second PCA component of pixel arrangements                            |
| thumb_pca_32               | float64 | Thirty-third image-based projection from PCA                                 |
| thumb_pca_33               | float64 | Thirty-fourth principal feature of thumbnail images                          |
| thumb_pca_34               | float64 | Thirty-fifth component reflecting saturation and hue variation               |
| thumb_pca_35               | float64 | Thirty-sixth visual component from thumbnail PCA                             |
| thumb_pca_36               | float64 | Thirty-seventh feature highlighting secondary visual traits                  |
| thumb_pca_37               | float64 | Thirty-eighth image PCA feature capturing edge detail                        |
| thumb_pca_38               | float64 | Thirty-ninth component encoding texture variation                            |
| thumb_pca_39               | float64 | Fortieth PCA direction of thumbnail pixel space                              |
| thumb_pca_40               | float64 | Forty-first image component derived from PCA analysis                        |
| thumb_pca_41               | float64 | Forty-second visual dimension summarizing pixel variance                     |
| thumb_pca_42               | float64 | Forty-third PCA feature capturing shadow or lighting effects                 |
| thumb_pca_43               | float64 | Forty-fourth visual pattern extracted via PCA                                |
| thumb_pca_44               | float64 | Forty-fifth component related to minor image details                         |
| thumb_pca_45               | float64 | Forty-sixth PCA projection of thumbnail pixel structure                      |
| thumb_pca_46               | float64 | Forty-seventh feature capturing texture and form                             |
| thumb_pca_47               | float64 | Forty-eighth component related to image granularity                          |
| thumb_pca_48               | float64 | Forty-ninth PCA projection of visual elements                                |
| thumb_pca_49               | float64 | Fiftieth component describing visual layout in thumbnails                    |
| thumb_pca_50               | float64 | Fifty-first PCA component from image features                                |
| thumb_pca_51               | float64 | Fifty-second image feature from PCA                                          |
| thumb_pca_52               | float64 | Fifty-third thumbnail feature capturing background patterns                  |
| thumb_pca_53               | float64 | Fifty-fourth component summarizing visual entropy                            |
| thumb_pca_54               | float64 | Fifty-fifth latent visual component                                          |
| thumb_pca_55               | float64 | Fifty-sixth PCA projection from pixel-level data                             |
| thumb_pca_56               | float64 | Fifty-seventh PCA component from image variance                              |
| thumb_pca_57               | float64 | Fifty-eighth visual feature from image PCA                                   |
| thumb_pca_58               | float64 | Fifty-ninth component describing non-obvious patterns                        |
| thumb_pca_59               | float64 | Sixtieth PCA feature derived from global color arrangements                  |
| thumb_pca_60               | float64 | Sixty-first PCA component encoding abstract visual patterns                  |
| thumb_pca_61               | float64 | Sixty-second image component from visual compression                         |
| thumb_pca_62               | float64 | Sixty-third PCA direction summarizing shape and color                        |
| thumb_pca_63               | float64 | Sixty-fourth visual feature from pixel representation                        |
| thumb_pca_64               | float64 | Sixty-fifth PCA projection highlighting aesthetic variance                   |
| thumb_pca_65               | float64 | Sixty-sixth thumbnail component reflecting brightness distribution           |
| thumb_pca_66               | float64 | Sixty-seventh image-based component from PCA                                 |
| thumb_pca_67               | float64 | Sixty-eighth visual PCA feature describing edge sharpness                    |
| thumb_pca_68               | float64 | Sixty-ninth component summarizing fine-grained image details                 |
| thumb_pca_69               | float64 | Seventieth component capturing background-foreground contrast                |
| thumb_pca_70               | float64 | Seventy-first PCA component capturing residual thumbnail variation           |
| thumb_pca_71               | float64 | Seventy-second latent feature of pixel structure                             |
| thumb_pca_72               | float64 | Seventy-third PCA component describing textural balance                      |
| thumb_pca_73               | float64 | Seventy-fourth component highlighting global image context                   |
| thumb_pca_74               | float64 | Seventy-fifth PCA direction reflecting lighting variance                     |
| thumb_pca_75               | float64 | Seventy-sixth image-based visual abstraction component                       |
| thumb_pca_76               | float64 | Seventy-seventh feature derived from subtle hue changes                      |
| thumb_pca_77               | float64 | Seventy-eighth component reflecting compositional elements                   |
| thumb_pca_78               | float64 | Seventy-ninth PCA projection of background textures                          |
| thumb_pca_79               | float64 | Eightieth visual component describing mid-level color regions                |
| thumb_pca_80               | float64 | Eighty-first PCA feature of thumbnail's image geometry                       |
| thumb_pca_81               | float64 | Eighty-second PCA direction capturing faint pattern distribution             |
| thumb_pca_82               | float64 | Eighty-third image abstraction from global thumbnail structure               |
| thumb_pca_83               | float64 | Eighty-fourth component derived from color blending                          |
| thumb_pca_84               | float64 | Eighty-fifth PCA feature related to structural emphasis                      |
| thumb_pca_85               | float64 | Eighty-sixth component capturing micro-patterns                              |
| thumb_pca_86               | float64 | Eighty-seventh feature summarizing peripheral image data                     |
| thumb_pca_87               | float64 | Eighty-eighth PCA direction of mid-frequency image signals                   |
| thumb_pca_88               | float64 | Eighty-ninth visual feature from thumbnail shape consistency                 |
| thumb_pca_89               | float64 | Ninetieth component encoding faint image textures                            |
| thumb_pca_90               | float64 | Ninety-first PCA projection of spatial arrangements                          |
| thumb_pca_91               | float64 | Ninety-second image component derived from noise-insensitive signals         |
| thumb_pca_92               | float64 | Ninety-third component summarizing low-contrast variance                     |
| thumb_pca_93               | float64 | Ninety-fourth PCA feature from background flattening patterns                |
| thumb_pca_94               | float64 | Ninety-fifth visual abstraction from complex pixel interactions              |
| thumb_pca_95               | float64 | Ninety-sixth PCA component encoding marginal image details                   |
| thumb_pca_96               | float64 | Ninety-seventh component from deep visual embeddings                         |
| thumb_pca_97               | float64 | Ninety-eighth latent visual component derived from color composition         |
| thumb_pca_98               | float64 | Ninety-ninth PCA projection describing fringe image structures               |
| thumb_pca_99               | float64 | One-hundredth PCA feature capturing fine residual patterns                   |