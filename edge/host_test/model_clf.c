#include <math.h>
#include <string.h>
static double sigmoid(double x) {
    if (x < 0.0) {
        double z = exp(x);
        return z / (1.0 + z);
    }
    return 1.0 / (1.0 + exp(-x));
}
void predict_knee_early(double * input, double * output) {
    double var0;
    if (input[7] < 10.559896) {
        if (input[11] < 0.000016646565) {
            var0 = 0.14630158;
        } else {
            var0 = -0.03821576;
        }
    } else {
        if (input[6] < 1.5637794) {
            var0 = -0.17740619;
        } else {
            var0 = -0.060123492;
        }
    }
    double var1;
    if (input[7] < 10.559896) {
        if (input[11] < 0.000016646565) {
            var1 = 0.13733554;
        } else {
            var1 = -0.035442982;
        }
    } else {
        var1 = -0.14338867;
    }
    double var2;
    if (input[7] < 10.559896) {
        if (input[11] < 0.000016646565) {
            var2 = 0.1298142;
        } else {
            var2 = -0.032892033;
        }
    } else {
        if (input[6] < 1.5637794) {
            var2 = -0.1501485;
        } else {
            var2 = -0.047247756;
        }
    }
    double var3;
    if (input[7] < 10.559896) {
        if (input[11] < 0.000016646565) {
            var3 = 0.123390935;
        } else {
            var3 = -0.030541033;
        }
    } else {
        if (input[6] < 1.5637794) {
            var3 = -0.13947959;
        } else {
            var3 = -0.044353418;
        }
    }
    double var4;
    if (input[7] < 10.559896) {
        if (input[11] < 0.000016646565) {
            var4 = 0.117818214;
        } else {
            var4 = -0.028371016;
        }
    } else {
        if (input[6] < 1.5637794) {
            var4 = -0.1305719;
        } else {
            var4 = -0.041664276;
        }
    }
    double var5;
    if (input[7] < 10.559896) {
        if (input[11] < 0.000016646565) {
            var5 = 0.112914495;
        } else {
            var5 = -0.026365388;
        }
    } else {
        if (input[6] < 1.5637794) {
            var5 = -0.12300187;
        } else {
            var5 = -0.0391609;
        }
    }
    double var6;
    if (input[5] < 413.5647) {
        if (input[3] < 0.000022163) {
            if (input[3] < -0.00052274) {
                var6 = 0.025937095;
            } else {
                var6 = 0.11095742;
            }
        } else {
            var6 = -0.016284814;
        }
    } else {
        if (input[11] < -0.00006655336) {
            var6 = 0.002021155;
        } else {
            var6 = -0.12669112;
        }
    }
    double var7;
    if (input[7] < 10.559896) {
        if (input[11] < 0.000016646565) {
            var7 = 0.10562771;
        } else {
            var7 = -0.023504077;
        }
    } else {
        if (input[10] < 607.7505) {
            var7 = -0.035994384;
        } else {
            var7 = -0.11463684;
        }
    }
    double var8;
    if (input[5] < 413.5647) {
        if (input[3] < 0.000022163) {
            if (input[11] < -0.00004145734) {
                var8 = 0.029935202;
            } else {
                var8 = 0.10339036;
            }
        } else {
            var8 = -0.014500201;
        }
    } else {
        if (input[11] < -0.00006655336) {
            var8 = 0.002308314;
        } else {
            var8 = -0.116075955;
        }
    }
    double var9;
    if (input[7] < 10.559896) {
        if (input[11] < 0.000016646565) {
            var9 = 0.09959745;
        } else {
            var9 = -0.020990042;
        }
    } else {
        if (input[10] < 607.7505) {
            var9 = -0.031350132;
        } else {
            var9 = -0.10558921;
        }
    }
    double var10;
    if (input[5] < 413.5647) {
        if (input[3] < 0.000022163) {
            if (input[3] < -0.000497063) {
                var10 = 0.026254883;
            } else {
                var10 = 0.09763659;
            }
        } else {
            var10 = -0.0129176965;
        }
    } else {
        if (input[11] < -0.00006655336) {
            var10 = 0.0025595336;
        } else {
            var10 = -0.10762326;
        }
    }
    double var11;
    if (input[7] < 10.559896) {
        if (input[11] < 0.000016646565) {
            var11 = 0.094400465;
        } else {
            var11 = -0.0187778;
        }
    } else {
        if (input[10] < 607.7505) {
            var11 = -0.027350042;
        } else {
            var11 = -0.09823912;
        }
    }
    double var12;
    if (input[5] < 413.5647) {
        if (input[3] < 0.000022163) {
            if (input[4] < 382.784) {
                var12 = 0.09279022;
            } else {
                var12 = 0.024911277;
            }
        } else {
            var12 = -0.01151381;
        }
    } else {
        if (input[11] < -0.00006655336) {
            var12 = 0.0027795534;
        } else {
            var12 = -0.10064751;
        }
    }
    double var13;
    if (input[9] < 0.15211079) {
        if (input[8] < 9.411096) {
            var13 = 0.06742262;
        } else {
            if (input[12] < 1.0842844) {
                var13 = -0.10906094;
            } else {
                var13 = -0.006386623;
            }
        }
    } else {
        if (input[1] < 0.01758214) {
            var13 = 0.020772606;
        } else {
            var13 = 0.09649804;
        }
    }
    double var14;
    if (input[5] < 413.5647) {
        if (input[3] < 0.000022163) {
            var14 = 0.078503944;
        } else {
            var14 = -0.010300033;
        }
    } else {
        if (input[11] < -0.00006655336) {
            var14 = 0.00390093;
        } else {
            var14 = -0.09424618;
        }
    }
    double var15;
    if (input[9] < 0.15211079) {
        if (input[8] < 9.411096) {
            var15 = 0.062903814;
        } else {
            if (input[12] < 1.0842844) {
                var15 = -0.102114156;
            } else {
                var15 = -0.0027834072;
            }
        }
    } else {
        if (input[8] < 9.745596) {
            var15 = 0.02182357;
        } else {
            var15 = 0.09113897;
        }
    }
    double var16;
    if (input[7] < 10.559896) {
        if (input[11] < 0.000016646565) {
            var16 = 0.085367896;
        } else {
            var16 = -0.014017937;
        }
    } else {
        if (input[10] < 607.7505) {
            var16 = -0.018801617;
        } else {
            var16 = -0.08504756;
        }
    }
    double var17;
    if (input[9] < 0.15211079) {
        if (input[8] < 9.457366) {
            var17 = 0.041195165;
        } else {
            if (input[12] < 1.0842844) {
                var17 = -0.10506572;
            } else {
                var17 = -0.0004404847;
            }
        }
    } else {
        if (input[8] < 9.745596) {
            var17 = 0.018175183;
        } else {
            var17 = 0.087050326;
        }
    }
    double var18;
    if (input[5] < 413.5647) {
        if (input[0] < 1.0787998) {
            var18 = 0.07483917;
        } else {
            var18 = -0.008920232;
        }
    } else {
        if (input[11] < -0.000030330353) {
            var18 = -0.002511613;
        } else {
            var18 = -0.08318549;
        }
    }
    double var19;
    if (input[9] < 0.15211079) {
        if (input[8] < 9.457366) {
            var19 = 0.037241302;
        } else {
            if (input[12] < 1.0842844) {
                var19 = -0.09989565;
            } else {
                var19 = 0.0019561804;
            }
        }
    } else {
        if (input[2] < 0.017301504) {
            var19 = 0.018152287;
        } else {
            var19 = 0.08149645;
        }
    }
    double var20;
    if (input[7] < 10.559896) {
        if (input[11] < 0.000016646565) {
            var20 = 0.07923668;
        } else {
            var20 = -0.013008283;
        }
    } else {
        if (input[10] < 607.7505) {
            var20 = -0.013411867;
        } else {
            var20 = -0.076564364;
        }
    }
    double var21;
    if (input[9] < 0.20851254) {
        if (input[7] < 10.342768) {
            if (input[9] < 0.03836429) {
                var21 = 0.07463535;
            } else {
                var21 = -0.035741493;
            }
        } else {
            var21 = -0.071831316;
        }
    } else {
        var21 = 0.07667713;
    }
    double var22;
    if (input[9] < 0.15211079) {
        if (input[7] < 10.342768) {
            if (input[9] < 0.0060297376) {
                var22 = 0.07026082;
            } else {
                var22 = -0.04370428;
            }
        } else {
            if (input[12] < 1.083945) {
                var22 = -0.079786725;
            } else {
                var22 = -0.021789204;
            }
        }
    } else {
        if (input[2] < 0.017357314) {
            var22 = 0.017108332;
        } else {
            var22 = 0.075596094;
        }
    }
    double var23;
    if (input[6] < 1.5637794) {
        var23 = -0.06075008;
    } else {
        if (input[3] < -0.000577019) {
            var23 = -0.028158976;
        } else {
            if (input[7] < 9.492166) {
                var23 = -0.0035086551;
            } else {
                var23 = 0.09955829;
            }
        }
    }
    double var24;
    if (input[9] < 0.20851254) {
        if (input[7] < 10.342768) {
            if (input[9] < 0.03836429) {
                var24 = 0.06922848;
            } else {
                var24 = -0.033836197;
            }
        } else {
            if (input[12] < 1.083945) {
                var24 = -0.07575916;
            } else {
                var24 = -0.019884117;
            }
        }
    } else {
        var24 = 0.071146436;
    }
    double var25;
    if (input[9] < 0.20851254) {
        if (input[7] < 10.342768) {
            if (input[9] < 0.03836429) {
                var25 = 0.06707929;
            } else {
                var25 = -0.03168775;
            }
        } else {
            if (input[12] < 1.083945) {
                var25 = -0.073294796;
            } else {
                var25 = -0.018977398;
            }
        }
    } else {
        var25 = 0.06898376;
    }
    double var26;
    if (input[6] < 1.5637794) {
        var26 = -0.05597051;
    } else {
        if (input[3] < -0.000577019) {
            var26 = -0.02524105;
        } else {
            if (input[7] < 9.492166) {
                var26 = -0.0053696535;
            } else {
                var26 = 0.09354726;
            }
        }
    }
    double var27;
    if (input[9] < 0.20851254) {
        if (input[7] < 10.342768) {
            if (input[3] < 0.000022163) {
                var27 = 0.06170986;
            } else {
                var27 = -0.0338397;
            }
        } else {
            var27 = -0.05928908;
        }
    } else {
        var27 = 0.06572259;
    }
    double var28;
    if (input[6] < 1.5637794) {
        var28 = -0.052252676;
    } else {
        if (input[3] < -0.000577019) {
            var28 = -0.023614462;
        } else {
            if (input[7] < 9.492166) {
                var28 = -0.0064667873;
            } else {
                var28 = 0.0902667;
            }
        }
    }
    double var29;
    if (input[9] < 0.20851254) {
        if (input[7] < 10.342768) {
            if (input[3] < 0.000022163) {
                var29 = 0.06003372;
            } else {
                var29 = -0.033210218;
            }
        } else {
            if (input[12] < 1.082137) {
                var29 = -0.06566339;
            } else {
                var29 = -0.017191293;
            }
        }
    } else {
        var29 = 0.06275644;
    }
    double var30;
    if (input[6] < 1.5637794) {
        var30 = -0.049404845;
    } else {
        if (input[3] < -0.000577019) {
            var30 = -0.021863861;
        } else {
            if (input[7] < 9.492166) {
                var30 = -0.007392816;
            } else {
                var30 = 0.086849354;
            }
        }
    }
    double var31;
    if (input[9] < 0.20851254) {
        if (input[8] < 9.457366) {
            var31 = 0.029067311;
        } else {
            if (input[12] < 1.0842844) {
                var31 = -0.06382368;
            } else {
                var31 = 0.009248217;
            }
        }
    } else {
        var31 = 0.060051035;
    }
    double var32;
    if (input[0] < 1.0787998) {
        if (input[9] < -0.09695427) {
            var32 = -0.018383507;
        } else {
            var32 = 0.064931296;
        }
    } else {
        if (input[5] < 382.6245) {
            var32 = 0.007232608;
        } else {
            var32 = -0.061652996;
        }
    }
    double var33;
    if (input[9] < 0.15211079) {
        if (input[3] < 0.000090351) {
            if (input[7] < 10.342768) {
                var33 = 0.0632446;
            } else {
                var33 = -0.04116841;
            }
        } else {
            var33 = -0.06940621;
        }
    } else {
        var33 = 0.04700574;
    }
    double var34;
    if (input[0] < 1.0787998) {
        if (input[9] < -0.09695427) {
            var34 = -0.015927864;
        } else {
            var34 = 0.06117654;
        }
    } else {
        if (input[5] < 382.6245) {
            var34 = 0.0054175877;
        } else {
            var34 = -0.057591733;
        }
    }
    double var35;
    if (input[3] < -0.000068115) {
        if (input[7] < 10.354321) {
            var35 = 0.06319574;
        } else {
            var35 = -0.012943109;
        }
    } else {
        if (input[2] < 0.017794212) {
            var35 = -0.06578606;
        } else {
            var35 = 0.017262323;
        }
    }
    double var36;
    if (input[0] < 1.0787998) {
        if (input[0] < 1.0693567) {
            var36 = -0.01809342;
        } else {
            var36 = 0.05641675;
        }
    } else {
        if (input[5] < 382.6245) {
            var36 = 0.005014062;
        } else {
            var36 = -0.054518837;
        }
    }
    double var37;
    if (input[3] < 0.000099076) {
        if (input[2] < 0.017228115) {
            if (input[11] < -0.00002401481) {
                var37 = 0.018424563;
            } else {
                var37 = -0.04756045;
            }
        } else {
            var37 = 0.07233551;
        }
    } else {
        var37 = -0.04856187;
    }
    double var38;
    if (input[9] < 0.15211079) {
        if (input[12] < 1.0842844) {
            if (input[8] < 9.577665) {
                var38 = -0.00023683603;
            } else {
                var38 = -0.06417984;
            }
        } else {
            var38 = 0.019565282;
        }
    } else {
        var38 = 0.041241717;
    }
    double var39;
    if (input[3] < -0.000068115) {
        if (input[7] < 10.354321) {
            var39 = 0.059474032;
        } else {
            var39 = -0.011893684;
        }
    } else {
        if (input[2] < 0.017794212) {
            var39 = -0.061798926;
        } else {
            var39 = 0.01474253;
        }
    }
    double var40;
    if (input[0] < 1.0787998) {
        if (input[0] < 1.0716245) {
            var40 = -0.01043169;
        } else {
            var40 = 0.0596563;
        }
    } else {
        if (input[7] < 9.523558) {
            var40 = -0.053286325;
        } else {
            var40 = 0.0042670737;
        }
    }
    double var41;
    if (input[3] < -0.000068115) {
        if (input[7] < 10.354321) {
            var41 = 0.057720125;
        } else {
            var41 = -0.011975593;
        }
    } else {
        if (input[2] < 0.017794212) {
            var41 = -0.059262134;
        } else {
            var41 = 0.01448103;
        }
    }
    double var42;
    if (input[9] < 0.15211079) {
        if (input[12] < 1.0842844) {
            var42 = -0.043015983;
        } else {
            var42 = 0.020384984;
        }
    } else {
        var42 = 0.03857865;
    }
    double var43;
    if (input[0] < 1.0787998) {
        if (input[0] < 1.0716245) {
            var43 = -0.009939085;
        } else {
            var43 = 0.05745806;
        }
    } else {
        if (input[6] < 10.605144) {
            var43 = -0.048596166;
        } else {
            var43 = 0.002587376;
        }
    }
    double var44;
    if (input[1] < 0.017164165) {
        var44 = -0.033806637;
    } else {
        if (input[3] < 0.000090351) {
            var44 = 0.04667831;
        } else {
            var44 = -0.016547015;
        }
    }
    double var45;
    if (input[3] < -0.000068115) {
        if (input[7] < 10.354321) {
            var45 = 0.054969102;
        } else {
            var45 = -0.010942012;
        }
    } else {
        if (input[2] < 0.017794212) {
            var45 = -0.0561839;
        } else {
            var45 = 0.013659609;
        }
    }
    double var46;
    if (input[0] < 1.0787998) {
        if (input[0] < 1.0716245) {
            var46 = -0.011215582;
        } else {
            var46 = 0.055571016;
        }
    } else {
        var46 = -0.028068656;
    }
    double var47;
    if (input[9] < 0.15211079) {
        if (input[10] < 608.8633) {
            var47 = 0.0070831864;
        } else {
            var47 = -0.04586672;
        }
    } else {
        var47 = 0.03604069;
    }
    double var48;
    if (input[3] < -0.000068115) {
        if (input[3] < -0.00031568) {
            var48 = -0.007929242;
        } else {
            var48 = 0.049550157;
        }
    } else {
        if (input[1] < 0.017655412) {
            var48 = -0.048973188;
        } else {
            var48 = 0.010100412;
        }
    }
    double var49;
    if (input[9] < 0.15211079) {
        if (input[10] < 608.8633) {
            var49 = 0.006378021;
        } else {
            var49 = -0.044069413;
        }
    } else {
        var49 = 0.034724366;
    }
    double var50;
    var50 = sigmoid(0.32277339226305085 + (var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12 + var13 + var14 + var15 + var16 + var17 + var18 + var19 + var20 + var21 + var22 + var23 + var24 + var25 + var26 + var27 + var28 + var29 + var30 + var31 + var32 + var33 + var34 + var35 + var36 + var37 + var38 + var39 + var40 + var41 + var42 + var43 + var44 + var45 + var46 + var47 + var48 + var49));
    memcpy(output, (double[]){1.0 - var50, var50}, 2 * sizeof(double));
}
