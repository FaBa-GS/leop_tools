cal0_1,rparam download 4 53                                 ,0,0,0,0,1627776240,0,0
cal0_2,rparam set euler_offset 0.0000 -0.0000 0.0000        ,0,0,0,0,1627776241,0,0
cal0_3,rparam set en_gso 0 0 1                              ,0,0,0,0,1627776242,0,0
cal0_4,rparam send                                          ,0,0,0,0,1627776243,0,0
cal0_5,adcs server 4 20                                     ,0,0,0,0,1627776244,0,0
cal0_6,adcs state setacs 6                                  ,0,0,0,0,1627776245,0,0
cal0_7,rparam download 4 89                                 ,0,0,0,0,1627776246,0,0
cal0_8,rparam set en_str 0                                  ,0,0,0,0,1627776247,0,0
cal0_9,rparam send                                          ,0,0,0,0,1627776248,0,0
cal0_10,power on 10 0 0 5428                                ,0,0,0,0,1627776259,0,0
cal0_11,rgosh server 4                                      ,0,0,0,0,1627776280,0,0
cal0_12,rgosh run 'st200 init 2'                            ,0,0,0,0,1627776281,0,0
cal0_13,rgosh run 'st200 custom "ping"'                     ,0,0,0,0,1627776285,0,0
cal0_14,rgosh run 'st200 custom "set keep-alive off"'       ,0,0,0,0,1627776289,0,0
cal0_15,rgosh run 'st200 custom "set param 32 1a 00 00 00 00 00 00"',0,0,0,0,1627776293,0,0
cal0_16,rgosh run 'st200 custom "set param 52 1a 00 00 00 00 00 00"',0,0,0,0,1627776297,0,0
cal0_17,rgosh run 'st200 custom "set param 53 80 01 00 00 00 00 00"',0,0,0,0,1627776301,0,0
cal0_18,rparam download 4 89                                ,0,0,0,0,1627776305,0,0
cal0_19,rparam set en_str 1                                 ,0,0,0,0,1627776306,0,0
cal0_20,rparam send                                         ,0,0,0,0,1627776307,0,0
cal0_21,hk server 1                                         ,0,0,0,0,1627781429,0,0
cal0_22,hk get 25 10 512 1627781428 /flash1/cal25_0.bin     ,0,0,0,0,1627781430,0,0
cal0_23,ftp server 1                                        ,0,0,0,0,1627781462,0,0
cal0_24,ftp zip /flash1/cal25_0.bin /flash1/cal25_0.zip     ,0,0,0,0,1627781463,0,0
cal0_25,rparam download 4 53                                ,0,0,0,0,1627781494,0,0
cal0_26,rparam set euler_offset 0.0000 0.0000 0.0000        ,0,0,0,0,1627781495,0,0
cal0_27,rparam set en_gso 0 0 1                             ,0,0,0,0,1627781496,0,0
cal0_28,rparam send                                         ,0,0,0,0,1627781497,0,0
cal0_29,adcs server 4 20                                    ,0,0,0,0,1627781498,0,0
cal0_30,adcs state setacs 6                                 ,0,0,0,0,1627781499,0,0
