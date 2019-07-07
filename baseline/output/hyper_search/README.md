Output results are saved here in `.csv` with naming scheme and columns
described below.

*Note: CSIFs are not known to be stable, so some output files might not be able
 to finish all of the rows; however, since random search is performed here,
rows are independent and lack of certain rows do not affect others. Row indices
 are also saved, so a search on the rows whose indices are missing from here
can be performed at a later time. Notice that doing so is no different from
performing a search on a newly generated `randparams*.pkl` .*

**Naming Scheme**:

`{start_row_index}_{end_row_index}.csv`


**Columns**:

`idx,acc,lr,batch_size,hl_acts,hl_sizes,decay,bNorm,dropout,reg`