# Aggregation Tree with Pseudo-Cube
---
Follow example.json to organize your database table structure into a configuration file，then run

``` 
python get_coreset.py coreset_size 0 config_file_address output_dir_name uniform
python get_coreset.py 1000 0 database_conf/yelp.json output_yelp 0
```

If you want to get coreset in batch， please run

``` sh
sh run.sh database_conf/yelp.json output_dir_name coreset_size_min coreset_size_max coreset_size_step time_min time_max uniform >> log_file
sh run.sh database_conf/yelp.json output_yelp 500 1000 100 1 5 0 >> yelp.log
```