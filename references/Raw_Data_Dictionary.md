# Raw Data Dictionary

The purpose of this file is to provide a clear and detailed description of the columns in the raw dataset. Here you will find definitions and explanations of the identified fields to facilitate understanding and analysis of the data, ensuring that all users have a consistent and coordinated view of the information contained.

This data dictionary was created using information provided at https://www.kaggle.com/datasets/canerkonuk/youtube-trending-videos-global/data

| Name                              | Type      | Description                                                                   |
| --------------------------------- | --------- | ----------------------------------------------------------------------------- | 
|video_id                           | object    | Unique identifier for the video on YouTube|
|video_published_at                 | object    | The date and time when the video was published|
|video_trending_date                | object    | The date when the video was identified as trending|
|video_trending_country             | object    | The country where the video is trending (ISO 3166-1 alpha-2 country code)|
|video_title                        | object    | The title of the video as displayed on YouTube|
|video_description                  | object    | The description provided by the video creator|
|video_default_thumbnail            | object    | URL of the default thumbnail for the video|
|video_category_id                  | object    | Numeric ID representing the category of the video|
|video_tags                         | object    | List of tags associated with the video for categorization and discoverability|
|video_duration                     | object    | Duration of the video in ISO 8601 format (e.g. "PT10M15S" for 10 minutes and 15 seconds)|
|video_dimension                    | object    | Dimension of the video (2d or 3d) |
|video_definition                   | object    | Video resolution quality (sd: Standard definition or hd - High definition) |
|video_licensed_content             | object    | Boolean indicating if the video contains licensed content|
|video_view_count                   | float64   | Total number of views for the video|
|video_like_count                   | float64   | Total number of likes for the video|
|video_comment_count                | float64   | Total number of comments on the video|
|channel_id                         | object    | Unique identifier for the YouTube channel|
|channel_title                      | object    | The name/title of the channel|
|channel_description                | object    | Description provided by the channel owner|
|channel_custom_url                 | object    | Custom URL for the channel (if available)|
|channel_published_at               | object    | The date and time when the channel was created|
|channel_country                    | object    | The country associated with the channel (if specified by the creator)|
|channel_view_count                 | float64   | Total number of views across all videos on the channel|
|channel_subscriber_count           | float64   | Total number of subscribers to the channel|
|channel_have_hidden_subscribers    | object    | Boolean indicating if the channel has hidden its subscriber count|
|channel_video_count                | float64   | Total number of videos uploaded by the channel|
|channel_localized_title            | object    | The localized title of the channel (if available in a different language)|
|channel_localized_description      | object    | The localized description of the channel (if available in a different language)| |