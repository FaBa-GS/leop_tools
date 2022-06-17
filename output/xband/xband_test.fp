xband0_1,rparam download 4 53                               ,0,0,0,0,1646783135,0,0
xband0_2,rparam set gso_ant 0.0000 0.0000 0.0000            ,0,0,0,0,1646783136,0,0
xband0_3,rparam set en_gso 1 0 1                            ,0,0,0,0,1646783137,0,0
xband0_4,rparam set gso_list 0.0 0.0 0.0                    ,0,0,0,0,1646783138,0,0
xband0_5,rparam send                                        ,0,0,0,0,1646783139,0,0
xband0_6,adcs server 4 20                                   ,0,0,0,0,1646783140,0,0
xband0_7,adcs state setacs 6                                ,0,0,0,0,1646783141,0,0
xband0_8,power on 9 6 0 1000                                ,0,0,0,0,1646784095,0,0
xband0_9,cmp clock sync 2                                   ,0,0,0,0,1646784126,0,0
xband0_10,power on 8 3 0 900                                ,0,0,0,0,1646784137,0,0
xband0_11,power on 6 5 0 850                                ,0,0,0,0,1646784168,0,0
xband0_12,power on 6 6 0 800                                ,0,0,0,0,1646784199,0,0
xband0_13,rparam download 17 11                             ,0,0,0,0,1646784230,0,0
xband0_14,rparam set modcod 13                              ,0,0,0,0,1646784231,0,0
xband0_15,rparam set symrate 30000000                       ,0,0,0,0,1646784232,0,0
xband0_16,rparam set roll_off 2                             ,0,0,0,0,1646784233,0,0
xband0_17,rparam set gain 64                                ,0,0,0,0,1646784234,0,0
xband0_18,rparam set enable true                            ,0,0,0,0,1646784235,0,0
xband0_19,rparam send                                       ,0,0,0,0,1646784236,0,0
xband0_20,rparam download 17 20                             ,0,0,0,0,1646784267,0,0
xband0_21,rparam set ensm_mode[1] fdd                       ,0,0,0,0,1646784268,0,0
xband0_22,rparam set fir_equalize[1] 1                      ,0,0,0,0,1646784269,0,0
xband0_23,rparam send                                       ,0,0,0,0,1646784270,0,0
xband0_24,rparam download 17 13                             ,0,0,0,0,1646784271,0,0
xband0_25,rparam set tx_mod 32apsk                          ,0,0,0,0,1646784272,0,0
xband0_26,rparam set tx_pwr_lvl 2                           ,0,0,0,0,1646784273,0,0
xband0_27,rparam set tx_freq 8100000000                     ,0,0,0,0,1646784274,0,0
xband0_28,rparam set dyn_if_adjust false                    ,0,0,0,0,1646784275,0,0
xband0_29,rparam set enable true                            ,0,0,0,0,1646784276,0,0
xband0_30,rparam send                                       ,0,0,0,0,1646784277,0,0
xband0_31,rparam download 17 20                             ,0,0,0,0,1646784278,0,0
xband0_32,rparam set tx_gain[1] -24.5                       ,0,0,0,0,1646784279,0,0
xband0_33,rparam send                                       ,0,0,0,0,1646784280,0,0
xband0_34,rparam download 17 13                             ,0,0,0,0,1646785574,0,0
xband0_35,rparam set enable false                           ,0,0,0,0,1646785575,0,0
xband0_36,rparam send                                       ,0,0,0,0,1646785576,0,0
xband0_37,power off 6 6                                     ,0,0,0,0,1646785607,0,0
xband0_38,power off 6 5                                     ,0,0,0,0,1646785638,0,0
xband0_39,z7000 node 2                                      ,0,0,0,0,1646785669,0,0
xband0_40,z7000 cmd exec /sbin/shutdown shutdown now        ,0,0,0,0,1646785670,0,0
xband0_41,power off 8 3                                     ,0,0,0,0,1646785701,0,0
xband0_42,power off 9 6                                     ,0,0,0,0,1646785732,0,0
xband0_43,rparam download 4 53                              ,0,0,0,0,1646785763,0,0
xband0_44,rparam set euler_offset 0.0000 0.0000 0.0000      ,0,0,0,0,1646785764,0,0
xband0_45,rparam set en_gso 0 0 1                           ,0,0,0,0,1646785765,0,0
xband0_46,rparam send                                       ,0,0,0,0,1646785766,0,0
xband0_47,adcs server 4 20                                  ,0,0,0,0,1646785767,0,0
xband0_48,adcs state setacs 6                               ,0,0,0,0,1646785768,0,0
xband0_49,hk server 1                                       ,0,0,0,0,1646785770,0,0
xband0_50,hk get 25 10 171 1646785769 /flash1/xband25_0.bin ,0,0,0,0,1646785771,0,0
xband0_51,ftp server 1                                      ,0,0,0,0,1646785803,0,0
xband0_52,ftp zip /flash1/xband25_0.bin /flash1/xband25_0.zip,0,0,0,0,1646785804,0,0
xband0_53,hk server 1                                       ,0,0,0,0,1646785836,0,0
xband0_54,hk get 22 10 171 1646785769 /flash1/xband22_0.bin ,0,0,0,0,1646785837,0,0
xband0_55,ftp server 1                                      ,0,0,0,0,1646785869,0,0
xband0_56,ftp zip /flash1/xband22_0.bin /flash1/xband22_0.zip,0,0,0,0,1646785870,0,0
xband0_57,hk server 1                                       ,0,0,0,0,1646785902,0,0
xband0_58,hk get 23 10 171 1646785769 /flash1/xband23_0.bin ,0,0,0,0,1646785903,0,0
xband0_59,ftp server 1                                      ,0,0,0,0,1646785935,0,0
xband0_60,ftp zip /flash1/xband23_0.bin /flash1/xband23_0.zip,0,0,0,0,1646785936,0,0
xband0_61,hk_srv beacon samplerate 25 high                  ,0,0,0,0,1646785967,0,0
