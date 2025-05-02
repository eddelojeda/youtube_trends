# Interim Data Dictionary

The purpose of this file is to provide a clear and detailed description of the columns in the interim processed dataset. Here you will find definitions and explanations of the identified fields to facilitate understanding and analysis of the data, ensuring that all users have a consistent and coordinated view of the information contained.

| Name                       | Type    | Description                                            |
| -------------------------- | ------- | ------------------------------------------------------ |
| video_published_at         | object  | The date and time when the video was published         |
| video_category_id          | object  | Numeric ID representing the category of the video      |
| video_duration             | object  | Duration of the video in ISO 8601 format               |
| video_view_count           | float64 | Total number of views for the video                    |
| video_like_count           | float64 | Total number of likes for the video                    |
| video_comment_count        | float64 | Total number of comments on the video                  |
| channel_custom_url         | object  | Custom URL for the channel (if available)              |
| channel_view_count         | float64 | Total number of views across all videos on the channel |
| channel_subscriber_count   | float64 | Total number of subscribers to the channel             |
| days_until_trend           | float64 | Total number of days until the video becomes a trend   |
| video_title_language       | object  | Language of the video title as displayed on YouTube    |
| video_title_translated     | object  | Translated title of the video                          |

From here on, each column is a binary (int64) flag representing whether at least one object of the specific class was detected in the thumbnail (0 = Not detected, 1 = Detected). 

| Name                       | Type  | Description                                                           |
| -------------------------- | ----- | --------------------------------------------------------------------- |
| thumbnail_person           | int64 | Indicator if an object of the class person was detected in thumbnail  |
| thumbnail_bicycle          | int64 | Indicator if an object of the class bicycle was detected in thumbnail |
| thumbnail_car              | int64 | Indicator if an object of the class car was detected in thumbnail     |
| thumbnail_motorcycle       | int64 | Indicator if an object of the class motorcycle was detected           |
| thumbnail_airplane         | int64 | Indicator if an object of the class airplane was detected             |
| thumbnail_bus              | int64 | Indicator if an object of the class bus was detected                  |
| thumbnail_train            | int64 | Indicator if an object of the class train was detected                |
| thumbnail_truck            | int64 | Indicator if an object of the class truck was detected                |
| thumbnail_boat             | int64 | Indicator if an object of the class boat was detected                 |
| thumbnail_traffic_light    | int64 | Indicator if an object of the class traffic_light was detected        |
| thumbnail_fire_hydrant     | int64 | Indicator if an object of the class fire_hydrant was detected         |
| thumbnail_stop_sign        | int64 | Indicator if an object of the class stop_sign was detected            |
| thumbnail_parking_meter    | int64 | Indicator if an object of the class parking_meter was detected        |
| thumbnail_bench            | int64 | Indicator if an object of the class bench was detected                |
| thumbnail_bird             | int64 | Indicator if an object of the class bird was detected                 |
| thumbnail_cat              | int64 | Indicator if an object of the class cat was detected                  |
| thumbnail_dog              | int64 | Indicator if an object of the class dog was detected                  |
| thumbnail_horse            | int64 | Indicator if an object of the class horse was detected                |
| thumbnail_sheep            | int64 | Indicator if an object of the class sheep was detected                |
| thumbnail_cow              | int64 | Indicator if an object of the class cow was detected                  |
| thumbnail_elephant         | int64 | Indicator if an object of the class elephant was detected             |
| thumbnail_bear             | int64 | Indicator if an object of the class bear was detected                 |
| thumbnail_zebra            | int64 | Indicator if an object of the class zebra was detected                |
| thumbnail_giraffe          | int64 | Indicator if an object of the class giraffe was detected              |
| thumbnail_backpack         | int64 | Indicator if an object of the class backpack was detected             |
| thumbnail_umbrella         | int64 | Indicator if an object of the class umbrella was detected             |
| thumbnail_handbag          | int64 | Indicator if an object of the class handbag was detected              |
| thumbnail_tie              | int64 | Indicator if an object of the class tie was detected                  |
| thumbnail_suitcase         | int64 | Indicator if an object of the class suitcase was detected             |
| thumbnail_frizbee          | int64 | Indicator if an object of the class frizbee was detected              |
| thumbnail_skis             | int64 | Indicator if an object of the class skis was detected                 |
| thumbnail_snowboard        | int64 | Indicator if an object of the class snowboard was detected            |
| thumbnail_sports_ball      | int64 | Indicator if an object of the class sports_ball was detected          |
| thumbnail_kite             | int64 | Indicator if an object of the class kite was detected                 |
| thumbnail_baseball_bat     | int64 | Indicator if an object of the class baseball_bat was detected         |
| thumbnail_baseball_glove   | int64 | Indicator if an object of the class baseball_glove was detected       |
| thumbnail_skateboard       | int64 | Indicator if an object of the class skateboard was detected           |
| thumbnail_surfboard        | int64 | Indicator if an object of the class surfboard was detected            |
| thumbnail_tennis_racket    | int64 | Indicator if an object of the class tennis_racket was detected        |
| thumbnail_bottle           | int64 | Indicator if an object of the class bottle was detected               |
| thumbnail_wine_glass       | int64 | Indicator if an object of the class wine_glass was detected           |
| thumbnail_cup              | int64 | Indicator if an object of the class cup was detected                  |
| thumbnail_fork             | int64 | Indicator if an object of the class fork was detected                 |
| thumbnail_knife            | int64 | Indicator if an object of the class knife was detected                |
| thumbnail_spoon            | int64 | Indicator if an object of the class spoon was detected                |
| thumbnail_bowl             | int64 | Indicator if an object of the class bowl was detected                 |
| thumbnail_banana           | int64 | Indicator if an object of the class banana was detected               |
| thumbnail_apple            | int64 | Indicator if an object of the class apple was detected                |
| thumbnail_sandwich         | int64 | Indicator if an object of the class sandwich was detected             |
| thumbnail_orange           | int64 | Indicator if an object of the class orange was detected               |
| thumbnail_broccoli         | int64 | Indicator if an object of the class broccoli was detected             |
| thumbnail_carrot           | int64 | Indicator if an object of the class carrot was detected               |
| thumbnail_hot_dog          | int64 | Indicator if an object of the class hot_dog was detected              |
| thumbnail_pizza            | int64 | Indicator if an object of the class pizza was detected                |
| thumbnail_donut            | int64 | Indicator if an object of the class donut was detected                |
| thumbnail_cake             | int64 | Indicator if an object of the class cake was detected                 |
| thumbnail_chair            | int64 | Indicator if an object of the class chair was detected                |
| thumbnail_couch            | int64 | Indicator if an object of the class couch was detected                |
| thumbnail_potted_plant     | int64 | Indicator if an object of the class potted_plant was detected         |
| thumbnail_bed              | int64 | Indicator if an object of the class bed was detected                  |
| thumbnail_dining_table     | int64 | Indicator if an object of the class dining_table was detected         |
| thumbnail_toilet           | int64 | Indicator if an object of the class toilet was detected               |
| thumbnail_tv               | int64 | Indicator if an object of the class tv was detected                   |
| thumbnail_laptop           | int64 | Indicator if an object of the class laptop was detected               |
| thumbnail_mouse            | int64 | Indicator if an object of the class mouse was detected                |
| thumbnail_remote           | int64 | Indicator if an object of the class remote was detected               |
| thumbnail_keyboard         | int64 | Indicator if an object of the class keyboard was detected             |
| thumbnail_cell_phone       | int64 | Indicator if an object of the class cell_phone was detected           |
| thumbnail_microwave        | int64 | Indicator if an object of the class microwave was detected            |
| thumbnail_oven             | int64 | Indicator if an object of the class oven was detected                 |
| thumbnail_toaster          | int64 | Indicator if an object of the class toaster was detected              |
| thumbnail_sink             | int64 | Indicator if an object of the class sink was detected                 |
| thumbnail_refrigerator     | int64 | Indicator if an object of the class refrigerator was detected         |
| thumbnail_book             | int64 | Indicator if an object of the class book was detected                 |
| thumbnail_clock            | int64 | Indicator if an object of the class clock was detected                |
| thumbnail_vase             | int64 | Indicator if an object of the class vase was detected                 |
| thumbnail_scissors         | int64 | Indicator if an object of the class scissors was detected             |
| thumbnail_teddy_bear       | int64 | Indicator if an object of the class teddy_bear was detected           |
| thumbnail_hair_drier       | int64 | Indicator if an object of the class hair_drier was detected           |
| thumbnail_toothbrush       | int64 | Indicator if an object of the class toothbrush was detected           |
