## Label Encoding Method

### Example: Trophic Mode


|             | Flag remove NA | Flag remain NA | Traditional Label Encode remove NA | Traditional Label Encode remain NA |
| ----------: | -------------- | -------------- | ---------------------------------- | ---------------------------------- |
|          na | 0(HCL prefer)  | 1              |                                    | 0                                  |
|  Pathotroph | 1              | 2              | 0                                  | 1                                  |
|  Saprotroph | 2              | 4              | 1                                  | 2                                  |
| Symbiotroph | 4              | 8              | 2                                  | 3                                  |



|                                   | Flag remove NA | Flag remain NA | Multi-Class remove NA | Multi-Class remain NA | Multi-Label remove NA (Pathotroph, Saprotroph, Symbiotroph) | Multi-Label remain NA (na,Pathotroph, Saprotroph, Symbiotroph) |
| --------------------------------- | -------------- | -------------- | --------------------- | --------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| na                                | 0(HCL prefer)  | 1              |                       | 0                     | 0(HCL prefer)                                               | (1,0,0,0)                                                    |
| Pathotroph                        | 1              | 2              | 0                     | 1                     | (1,0,0)                                                     | (0,1,0,0)                                                    |
| Pathotroph-Symbiotroph            | 1+4=5          | 2+8=10         | 1                     | 2                     | (1,0,1)                                                     | (0,1,0,1)                                                    |
| Pathotroph-Saprotroph             | 1+2=3          | 2+4=6          | 2                     | 3                     | (1,1,0)                                                     | (0,1,1,0)                                                    |
| Pathotroph-Saprotroph-Symbiotroph | 1+2+4=7        | 2+4+8=14       | 3                     | 4                     | (1,1,1)                                                     | (0,1,1,1)                                                    |
| Saprotroph                        | 2              | 4              | 4                     | 5                     | (0,1,0)                                                     | (0,0,1,0)                                                    |
| Saprotroph-Symbiotroph            | 2+4=6          | 4+8=12         | 5                     | 6                     | (0,1,1)                                                     | (0,0,1,1)                                                    |
| Symbiotroph                       | 4              | 8              | 6                     | 7                     | (0,0,1)                                                     | (0,0,0,1)                                                    |

1. Flag encoding 方式沒有零 (Zero-Truncated Model)
2. SKlearn Package 中 Multi-Class ALG 是否支援不連續整數的分類 ?
3. PyTorch 基本NN ALG 是否支援不連續整數的分類 ?



