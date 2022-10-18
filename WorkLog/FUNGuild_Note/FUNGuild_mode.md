### Definition From FUNGuild Paper Description

#### Trophic_Mode:

We have organized entries into three broad groupings refered to as trophic modes (sensu Tedersoo et al., 2014):
(1) pathotroph = receiving nutrients by harming host cells (including phagotrophs); 
(2) symbiotroph = receiving nutrients by exchanging resources with host cells; and 
(3) saprotroph = receiving nutrients by breaking down dead host cells. 

```tex
Pathotroph
Symbiotroph
Saprotroph
```

#### 以下是 GUILD MODE 真正出現的LABLE
```tex
Saprotroph
Pathotroph-Saprotroph-Symbiotroph
na
Symbiotroph
Pathotroph
Pathotroph-Symbiotroph
Pathotroph-Saprotroph
Saprotroph-Symbiotroph
```

#### Guild_Mode:

While these trophic definitions may differ among fields (e.g. pathology vs. ecology), we think that these broadly defined trophic modes work well in fungal community ecology as they reflect the major feeding habits of fungi. 
Within these trophic modes, we designated a total of 12 categories broadly refered to as guilds in alphabetical order: 

- animal pathogens
- arbuscular mycorrhizal fungi
- ectomycorrhizal fungi
- ericoid mycorrhizal fungi
- foliar endophytes
- lichenicolous fungi
- lichenized fungi
- mycoparasites
- plant pathogens
- undefined root endophytes
- undefined saprotrophs
- wood saprotrophs

### [FUNGuild GitHub Description ](https://github.com/UMNFuN/FUNGuild):

Provide a relevant guild descriptor

| Tropic mode | Guild Mode                 | Note                                                         |
| ----------- | -------------------------- | ------------------------------------------------------------ |
| Pathotroph  | Animal Pathogen            | 1. 動物病原菌<br />2. including human pathogens - typically annoated as such |
|             | Bryophyte Parasite         | 1. 苔蘚植物寄生菌                                            |
|             | Clavicipitaceous Endophyte | 1. 內生真菌<br />2. Several Clavicipitaceous endophytes produce compounds that inhibit the growth of other fungi in vitro |
|             | Fungal Parasite            | 1. 真菌寄生菌                                                |
|             | Plant Pathogen             | 1.高等植物病原菌<br />2. Appeared at Updated Database but Not in Discussion Before |
|             | Lichen Parasite            | 1. Appeared at Updated Database but Not in Discussion Before |
|             | Bryophyte Parasite         | 1. Appeared at Updated Database but Not in Discussion Before |
|             | Insect Pathogen            | 1. Appeared at Updated Database but Not in Discussion Before |
| Saprotroph  | Dung Saprotroph            | 1. 排泄物腐生菌<br />2. i.e., coprophilous                   |
|             | Leaf Saprotroph            | 1. 葉子腐生菌<br />2. e.g., leaf litter decomposer           |
|             | Plant Saprotroph           | 1. 腐敗植物腐生菌                                            |
|             | Soil Saprotroph            | 1. 土壤腐生菌<br />2. e.g., rhizosphere saprobe - typically annoated as a rhizosphere fungus |
|             | Undefined Saprotroph       | 2. e.g., a general saprobe, or in cases where the ecology is not known but suspected to be a saprobe) |
|             | Wood Saprotroph            | 1. 木質腐生菌<br />2. e.g., wood rotting fungi)              |
|             | Litter Saprotroph          | 1. Appeared at Updated Database but Not in Discussion Before |
| Symbiotroph | Ectomycorrhizal            | 1. 外生菌根                                                  |
|             | Ericoid Mycorrhizal        | 1. 杜鵑花類菌根                                              |
|             | Arbuscular Mycorrhizal     | 1. 叢枝菌根菌<br />2. appeared at Updated Database but Not in Discussion Before |
|             | Orchid Mycorrhizal         | 1. Appeared at Updated Database but Not in Discussion Before |
|             | Endophyte                  | 1. Any organism that lives inside another plantSynonyms      |
|             | Epiphyte                   | 1. A plant that grows on another, using it for physical support but obtaining no nutrients from it and neither causing damage nor offering benefit; an air plant. |
|             | Lichenized                 | 1. 地衣共生菌<br />2. i.e., lichen                           |
|             | Animal Endosymbiont        | 1. It is related to symbionts according to Wiki <br />2. Appeared at Updated Database but Not in Discussion Before |
|             |                            |                                                              |
```tex
Animal Pathogen
Bryophyte Parasite
Clavicipitaceous Endophyte
Fungal Parasite
Plant Pathogen
Lichen Parasite
Bryophyte Parasite
Insect Pathogen
Dung Saprotroph
Leaf Saprotroph 
Plant Saprotroph
Soil Saprotroph
Undefined Saprotroph
Wood Saprotroph
Litter Saprotroph
Ectomycorrhiza
Ericoid Mycorrhizal
Arbuscular Mycorrhizal
Orchid Mycorrhizal
Endophyte 
Epiphyte
Lichenized
Animal Endosymbiont
```

#### 以下是 GUILD MODE 真正出現的LABLE
```text
Animal Endosymbiont
Animal Endosymbiont-Animal Pathogen-Endophyte-Plant Pathogen-Undefined Saprotroph
Animal Endosymbiont-Animal Pathogen-Plant Pathogen-Undefined Saprotroph
Animal Endosymbiont-Animal Pathogen-Undefined Saprotroph
Animal Endosymbiont-Plant Saprotroph
Animal Endosymbiont-Undefined Saprotroph
Animal Pathogen
Animal Pathogen-Clavicipitaceous Endophyte-Fungal Parasite
Animal Pathogen-Dung Saprotroph
Animal Pathogen-Endophyte-Epiphyte-Fungal Parasite-Plant Pathogen-Wood Saprotroph
Animal Pathogen-Endophyte-Epiphyte-Plant Pathogen
Animal Pathogen-Endophyte-Epiphyte-Plant Pathogen-Undefined Saprotroph
Animal Pathogen-Endophyte-Epiphyte-Undefined Saprotroph
Animal Pathogen-Endophyte-Fungal Parasite-Plant Pathogen-Wood Saprotroph
Animal Pathogen-Endophyte-Lichen Parasite-Plant Pathogen-Soil Saprotroph-Wood Saprotroph
Animal Pathogen-Endophyte-Plant Pathogen
Animal Pathogen-Endophyte-Plant Pathogen-Wood Saprotroph
Animal Pathogen-Endophyte-Plant Saprotroph-Soil Saprotroph
Animal Pathogen-Endophyte-Undefined Saprotroph
Animal Pathogen-Endophyte-Wood Saprotroph
Animal Pathogen-Fungal Parasite-Undefined Saprotroph
Animal Pathogen-Plant Pathogen
Animal Pathogen-Soil Saprotroph
Animal Pathogen-Undefined Saprotroph
Arbuscular Mycorrhizal
Bryophyte Parasite-Leaf Saprotroph-Soil Saprotroph-Undefined Saprotroph-Wood Saprotroph
Dung Saprotroph
Dung Saprotroph-Endophyte-Epiphyte-Wood Saprotroph
Dung Saprotroph-Undefined Saprotroph
Ectomycorrhizal
Ectomycorrhizal-Wood Saprotroph
Endophyte
Endophyte-Fungal Parasite-Plant Pathogen
Endophyte-Insect Pathogen
Endophyte-Lichen Parasite-Plant Pathogen
Endophyte-Litter Saprotroph-Soil Saprotroph-Undefined Saprotroph
Endophyte-Plant Pathogen
Endophyte-Plant Pathogen-Undefined Saprotroph
Endophyte-Plant Pathogen-Wood Saprotroph
Endophyte-Undefined Saprotroph
Epiphyte
Epiphyte-Plant Pathogen-Wood Saprotroph
Ericoid Mycorrhizal
Fungal Parasite
Fungal Parasite-Plant Pathogen-Plant Saprotroph
Lichen Parasite
Lichenized
na
Orchid Mycorrhizal
Plant Pathogen
Plant Pathogen-Undefined Saprotroph
Plant Pathogen-Wood Saprotroph
Plant Saprotroph
Plant Saprotroph-Wood Saprotroph
Soil Saprotroph
Undefined Saprotroph
Undefined Saprotroph-Wood Saprotroph
Wood Saprotroph
```


