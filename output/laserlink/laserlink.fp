laser_op0_1,rparam download 4 53                            ,0,0,0,0,1629450600,0,0
laser_op0_2,rparam set gso_ant 1.5708 0.0000 -1.5708        ,0,0,0,0,1629450601,0,0
laser_op0_3,rparam set en_gso 1 0 1                         ,0,0,0,0,1629450602,0,0
laser_op0_4,rparam set gso_list 4206093.0 824082.0 4708434.0,0,0,0,0,1629450603,0,0
laser_op0_5,rparam send                                     ,0,0,0,0,1629450604,0,0
laser_op0_6,adcs server 4 20                                ,0,0,0,0,1629450605,0,0
laser_op0_7,adcs state setacs 6                             ,0,0,0,0,1629450606,0,0
laser_op0_8,eps node 2                                      ,0,0,0,0,1629451607,0,0
laser_op0_9,eps output 3 0 1000                             ,0,0,0,0,1629451608,0,0
laser_op0_10,eps output 3 1 0                               ,0,0,0,0,1629451609,0,0
laser_op0_11,eps output 2 0 1000                            ,0,0,0,0,1629451640,0,0
laser_op0_12,eps output 2 1 0                               ,0,0,0,0,1629451641,0,0
laser_op0_13,eps output 7 0 1000                            ,0,0,0,0,1629451672,0,0
laser_op0_14,eps output 7 1 0                               ,0,0,0,0,1629451673,0,0
laser_op0_15,cmp clock sync 7                               ,0,0,0,0,1629451704,0,0
laser_op0_16,rparam download 8 1                            ,0,0,0,0,1629451735,0,0
laser_op0_17,rparam set ant-rx-on 1 1 1 1                   ,0,0,0,0,1629451736,0,0
laser_op0_18,rparam send                                    ,0,0,0,0,1629451737,0,0
laser_op0_19,rparam download 8 5                            ,0,0,0,0,1629451738,0,0
laser_op0_20,rparam set ant-tx-pwr 1 1 1 1                  ,0,0,0,0,1629451739,0,0
laser_op0_21,rparam set ant-tx-en 1 1 1 1                   ,0,0,0,0,1629451740,0,0
laser_op0_22,rparam set ant-tx-on 1 1 1 1                   ,0,0,0,0,1629451741,0,0
laser_op0_23,rparam set gain -50                            ,0,0,0,0,1629451742,0,0
laser_op0_24,rparam send                                    ,0,0,0,0,1629451743,0,0
laser_op0_25,eps output 3 0 0                               ,0,0,0,0,1629452400,0,0
laser_op0_26,eps output 2 0 0                               ,0,0,0,0,1629452431,0,0
laser_op0_27,eps output 7 0 0                               ,0,0,0,0,1629452462,0,0
laser_op0_28,rparam download 4 53                           ,0,0,0,0,1629452493,0,0
laser_op0_29,rparam set euler_offset 1.5708 0.0000 -1.5708  ,0,0,0,0,1629452494,0,0
laser_op0_30,rparam set en_gso 0 0 1                        ,0,0,0,0,1629452495,0,0
laser_op0_31,rparam send                                    ,0,0,0,0,1629452496,0,0
laser_op0_32,adcs server 4 20                               ,0,0,0,0,1629452497,0,0
laser_op0_33,adcs state setacs 6                            ,0,0,0,0,1629452498,0,0
laser_op0_34,hk server 1                                    ,0,0,0,0,1629452500,0,0
laser_op0_35,hk get 22 10 89 1629452493 /flash1/laser_op22_0.bin,0,0,0,0,1629452501,0,0
laser_op0_36,ftp server 1                                   ,0,0,0,0,1629452533,0,0
laser_op0_37,ftp zip /flash1/laser_op22_0.bin /flash1/laser_op22_0.zip,0,0,0,0,1629452534,0,0
laser_op0_38,hk server 1                                    ,0,0,0,0,1629452566,0,0
laser_op0_39,hk get 23 10 89 1629452493 /flash1/laser_op23_0.bin,0,0,0,0,1629452567,0,0
laser_op0_40,ftp server 1                                   ,0,0,0,0,1629452599,0,0
laser_op0_41,ftp zip /flash1/laser_op23_0.bin /flash1/laser_op23_0.zip,0,0,0,0,1629452600,0,0
laser_op0_42,hk server 1                                    ,0,0,0,0,1629452632,0,0
laser_op0_43,hk get 25 10 89 1629452493 /flash1/laser_op25_0.bin,0,0,0,0,1629452633,0,0
laser_op0_44,ftp server 1                                   ,0,0,0,0,1629452665,0,0
laser_op0_45,ftp zip /flash1/laser_op25_0.bin /flash1/laser_op25_0.zip,0,0,0,0,1629452666,0,0
laser_op0_46,hk server 1                                    ,0,0,0,0,1629452698,0,0
laser_op0_47,hk get 31 30 30 1629452493 /flash1/laser_op31_0.bin,0,0,0,0,1629452699,0,0
laser_op0_48,ftp server 1                                   ,0,0,0,0,1629452731,0,0
laser_op0_49,ftp zip /flash1/laser_op31_0.bin /flash1/laser_op31_0.zip,0,0,0,0,1629452732,0,0
