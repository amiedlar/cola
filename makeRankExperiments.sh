preparedata load mg_scale \
    replace-column --scale-col 0 --scale-by 1.1 5 scale \
    replace-column --scale-col 1 --scale-by .9  4 scale \
    replace-column --weights "-0.1 -0.2 2 0 0" 3 weights \
    replace-column 2 uniform \
    split

preparedata load mg_scale \
    insert-column --scale-col 0 --scale-by 1.1 scale \
    insert-column --scale-col 1 --scale-by 0.9 scale \
    insert-column --weights "1 2 3 4 5 6" weights \
    insert-column uniform \
    split
