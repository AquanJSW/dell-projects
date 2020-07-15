@ echo off
%1 %2
ver|find "5.">nul&&goto :Admin
mshta vbscript:createobject("shell.application").shellexecute("%~s0","goto :Admin","","runas",1)(window.close)&goto :eof
:Admin

route delete 0.0.0.0
route add 0.0.0.0 mask 0.0.0.0 192.168.43.216
echo Success!
