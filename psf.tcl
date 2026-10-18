package require autopsf
puts "hello from script"
#mol new [lindex $argv 1]
autopsf -mol 0 -protein -regen
exit
