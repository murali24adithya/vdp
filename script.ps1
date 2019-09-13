
$instance = $args[0]
$trainfile = "./concepts/"+ $instance + "/" + $instance + "_train_out.txt"
$testfile = "./concepts/"+ $instance + "/" + $instance + "_test_out.txt"

$num_vars = $args[1]
$num_rels = $args[2]

if($args[3] -eq $null)
{$if_existential = 'no'}
else
{$if_existential = $args[3]}

if($args[4] -eq $null)
{$if_conjuncts = 'no'}
else
{$if_conjuncts = $args[4]}


py main.py $trainfile $testfile $num_vars $num_rels $if_existential $if_conjuncts 





