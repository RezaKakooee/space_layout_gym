docker run --gpus='"device=0"' -d -it -p 9999:9999 -v /home/iakakooe/dbt/housing_design:/app/main --name="dbt_housing" dbt_housing bash
#docker run --gpus='"device=0"' -d -it -p 9999:9999 -v /home/iakakooe/dbt/housing_design:/app/main --name="dbt_housing_name" dbt_housing_img bash
