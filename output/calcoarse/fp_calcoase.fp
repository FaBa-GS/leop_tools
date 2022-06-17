cal0_1,adcs server 4 20                                     ,0,0,0,0,1629451800,0,0
cal0_2,adcs state setephem 1                                ,0,0,0,0,1629451801,0,0
cal0_3,adcs state setads 3                                  ,0,0,0,0,1629451802,0,0
cal0_4,adcs server 4 20                                     ,0,0,0,0,1629451842,0,0
cal0_5,adcs state setacs 2                                  ,0,0,0,0,1629451843,0,0
cal0_6,rparam download 4 50                                 ,0,0,0,0,1629451848,0,0
cal0_7,rparam set bdot_axis 1                               ,0,0,0,0,1629451849,0,0
cal0_8,rparam set bdot_sign 1                               ,0,0,0,0,1629451850,0,0
cal0_9,rparam send                                          ,0,0,0,0,1629451851,0,0
cal0_10,rparam download 4 50                                ,0,0,0,0,1629453052,0,0
cal0_11,rparam set bdot_axis 1                              ,0,0,0,0,1629453053,0,0
cal0_12,rparam set bdot_sign -1                             ,0,0,0,0,1629453054,0,0
cal0_13,rparam send                                         ,0,0,0,0,1629453055,0,0
cal0_14,rparam download 4 50                                ,0,0,0,0,1629454256,0,0
cal0_15,rparam set bdot_axis 0                              ,0,0,0,0,1629454257,0,0
cal0_16,rparam set bdot_sign -1                             ,0,0,0,0,1629454258,0,0
cal0_17,rparam send                                         ,0,0,0,0,1629454259,0,0
cal0_18,rparam download 4 50                                ,0,0,0,0,1629455460,0,0
cal0_19,rparam set bdot_axis 0                              ,0,0,0,0,1629455461,0,0
cal0_20,rparam set bdot_sign 1                              ,0,0,0,0,1629455462,0,0
cal0_21,rparam send                                         ,0,0,0,0,1629455463,0,0
cal0_22,rparam download 4 50                                ,0,0,0,0,1629456664,0,0
cal0_23,rparam set bdot_axis 2                              ,0,0,0,0,1629456665,0,0
cal0_24,rparam set bdot_sign -1                             ,0,0,0,0,1629456666,0,0
cal0_25,rparam send                                         ,0,0,0,0,1629456667,0,0
cal0_26,rparam download 4 50                                ,0,0,0,0,1629457868,0,0
cal0_27,rparam set bdot_axis 2                              ,0,0,0,0,1629457869,0,0
cal0_28,rparam set bdot_sign 1                              ,0,0,0,0,1629457870,0,0
cal0_29,rparam send                                         ,0,0,0,0,1629457871,0,0
cal0_30,hk server 1                                         ,0,0,0,0,1629459073,0,0
cal0_31,hk get 25 10 726 1629459072 /flash1/cal0.bin        ,0,0,0,0,1629459074,0,0
cal0_32,ftp server 1                                        ,0,0,0,0,1629459106,0,0
cal0_33,ftp zip /flash1/cal0.bin /flash1/cal0.zip           ,0,0,0,0,1629459107,0,0
