#!/bin/sh

display_usage() { 
	echo "Usage: $0 [eventlist folder]" 
}

if [ $# -lt 1 ]; then
	display_usage
	exit 1
fi 


fill='0\t2\t\"\"\t\"\"\t0\t0\t0\t0\t0\t0\t0\n'

# delete extra boundaries
perl -i.bak -ne 'print unless $. == 3760' $1/elist_6.csv
perl -i.bak -ne 'print unless $. == 2435' $1/elist_14.csv
perl -i.bak -ne 'print unless $. == 2503 || $. == 3395 || $. == 3661 || $. == 3738' $1/elist_12.csv
perl -i.bak -ne 'print unless $. == 543' $1/elist_23.csv

# add missing story codes
for sid in 16 28 30 40; do
	perl -i.bak -pe "print \"${fill}\" if \$. == 3" $1/elist_${sid}.csv
done
