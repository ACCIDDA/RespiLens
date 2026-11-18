export const targetDisplayNameMap = {
  "wk inc covid hosp": "Weekly Incident COVID-19 Hospitalizations",
  "wk inc covid prop ed visits": "Proportion of ED Visits due to COVID-19",
  "wk inc flu hosp": "Weekly Incident Flu Hospitalizations",
  "wk inc flu prop ed visits": "Proportion of ED Visits due to Flu",
  "wk inc rsv hosp": "Weekly Incident RSV Hospitalizations",
  "wk inc rsv prop ed visits": "Proportion of ED Visits due to RSV",
  "Flu ED visits pct": "Percent of ED Visits due to Flu",
  "ILI ED visits pct": "Percent of ED Visits due to Influenza-like Illness",
  "COVID ED visits pct": "Percent of ED Visits due to COVID-19",
  "RSV ED visits pct": "Percent of ED Visits due to RSV",
  "Flu hosp admits": "Weekly Flu Hospital Admissions",
};

export const targetYAxisLabelMap = {
  "Weekly Incident COVID-19 Hospitalizations": "COVID-19 Hospitalizations",
  "Proportion of ED Visits due to COVID-19":
    "Proportion of ED Visits due to COVID-19",
  "Weekly Incident Flu Hospitalizations": "Flu Hospitalizations",
  "Proportion of ED Visits due to Flu": "Proportion of ED Visits due to Flu",
  "Weekly Incident RSV Hospitalizations": "RSV Hospitalizations",
  "Proportion of ED Visits due to RSV": "Proportion of ED Visits due to RSV",
  "Percent of ED Visits due to Flu": "% of ED Visits",
  "Percent of ED Visits due to Influenza-like Illness": "% of ED Visits",
  "Percent of ED Visits due to COVID-19": "% of ED Visits",
  "Percent of ED Visits due to RSV": "% of ED Visits",
  "Weekly Flu Hospital Admissions": "Flu Hospital Admissions",
};

export const nhsnTargetsToColumnsMap = {
  "Hospital Admissions (count)": [
    "Total Adult COVID-19 Admissions",
    "Total Adult Influenza Admissions",
    "Total Adult RSV Admissions",
    "Total COVID-19 Admissions",
    "Total ICU Patients Hospitalized with COVID-19 ",
    "Total ICU Patients Hospitalized with Influenza",
    "Total ICU Patients Hospitalized with RSV",
    "Total Influenza Admissions",
    "Total Patients Hospitalized with COVID-19",
    "Total Patients Hospitalized with Influenza",
    "Total Patients Hospitalized with RSV",
    "Total Pediatric COVID-19 Admissions",
    "Total Pediatric Influenza Admissions",
    "Total Pediatric RSV Admissions",
    "Total RSV Admissions",
    "Number of Adult COVID-19 Admissions, 18-49 years",
    "Number of Adult COVID-19 Admissions, 50-64 years",
    "Number of Adult COVID-19 Admissions, 65-74 years",
    "Number of Adult COVID-19 Admissions, 75+ years",
    "Number of Adult ICU Patients Hospitalized with COVID-19",
    "Number of Adult ICU Patients Hospitalized with Influenza",
    "Number of Adult ICU Patients Hospitalized with RSV",
    "Number of Adult Influenza Admissions, 18-49 years",
    "Number of Adult Influenza Admissions, 50-64 years",
    "Number of Adult Influenza Admissions, 65-74 years",
    "Number of Adult Influenza Admissions, 75+ years",
    "Number of Adult Patients Hospitalized with COVID-19",
    "Number of Adult Patients Hospitalized with Influenza",
    "Number of Adult Patients Hospitalized with RSV",
    "Number of Adult RSV Admissions, 18-49 years",
    "Number of Adult RSV Admissions, 50-64 years",
    "Number of Adult RSV Admissions, 65-74 years",
    "Number of Adult RSV Admissions, 75+ years",
    "Number of COVID-19 Admissions, unknown age",
    "Number of Influenza Admissions, unknown age",
    "Number of Pedatric Patients Hospitalized with Influenza",
    "Number of Pediatric COVID-19 Admissions, 0-4 years",
    "Number of Pediatric COVID-19 Admissions, 5-17 years",
    "Number of Pediatric ICU Patients Hospitalized with COVID-19",
    "Number of Pediatric ICU Patients Hospitalized with Influenza",
    "Number of Pediatric ICU Patients Hospitalized with RSV",
    "Number of Pediatric Influenza Admissions, 0-4 years",
    "Number of Pediatric Influenza Admissions, 5-17 years",
    "Number of Pediatric Patients Hospitalized with COVID-19",
    "Number of Pediatric Patients Hospitalized with RSV",
    "Number of Pediatric RSV Admissions, 0-4 years",
    "Number of Pediatric RSV Admissions, 5-17 years",
    "Number of RSV Admissions, unknown age",
  ],
  "Hospital Admissions (rate)": [
    "Total Number of Adult COVID-19 Admissions per 100,000 population",
    "Total Number of Adult Influenza Admissions per 100,000 population",
    "Total Number of Adult RSV Admissions per 100,000 population",
    "Total Number of Pediatric COVID-19 Admissions per 100,000 population",
    "Total Number of Pediatric Influenza Admissions per 100,000 population",
    "Total Number of Pediatric RSV Admissions per 100,000 population",
    "Total number of COVID-19 Admissions per 100,000 population",
    "Total number of Influenza Admissions per 100,000 population",
    "Total number of RSV Admissions per 100,000 population",
    "Number of Adult COVID-19 Admissions, 18-49 years, per 100,000 population",
    "Number of Adult COVID-19 Admissions, 50-64 years, per 100,000 population",
    "Number of Adult COVID-19 Admissions, 65-74 years, per 100,000 population",
    "Number of Adult COVID-19 Admissions, 75+ years, per 100,000 population",
    "Number of Adult Influenza Admissions, 18-49 years, per 100,000 population",
    "Number of Adult Influenza Admissions, 50-64 years, per 100,000 population",
    "Number of Adult Influenza Admissions, 65-74 years, per 100,000 population",
    "Number of Adult Influenza Admissions, 75+ years, per 100,000 population",
    "Number of Adult RSV Admissions, 18-49 years, per 100,000 population",
    "Number of Adult RSV Admissions, 50-64 years, per 100,000 population",
    "Number of Adult RSV Admissions, 65-74 years, per 100,000 population",
    "Number of Adult RSV Admissions, 75+ years, per 100,000 population",
    "Number of Pediatric COVID-19 Admissions, 0-4 years, per 100,000 population",
    "Number of Pediatric COVID-19 Admissions, 5-17 years, per 100,000 population",
    "Number of Pediatric Influenza Admissions, 0-4 years, per 100,000 population",
    "Number of Pediatric Influenza Admissions, 5-17 years, per 100,000 population",
    "Number of Pediatric RSV Admissions, 0-4 years, per 100,000 population",
    "Number of Pediatric RSV Admissions, 5-17 years, per 100,000 population",
  ],
  "Hospital Admissions (%)": [
    "Percent Adult COVID-19 Admissions",
    "Percent Adult Influenza Admissions",
    "Percent Adult RSV Admissions",
    "Percent Pediatric COVID-19 Admissions",
    "Percent Pediatric Influenza Admissions",
    "Percent Pediatric RSV Admissions",
  ],
  "Bed Capacity (count)": [
    "Number of Adult ICU Beds",
    "Number of Adult ICU Beds Occupied",
    "Number of Adult Inpatient Beds",
    "Number of Adult Inpatient Beds Occupied",
    "Number of ICU Beds",
    "Number of ICU Beds Occupied",
    "Number of Inpatient Beds",
    "Number of Inpatient Beds Occupied",
    "Number of Pediatric ICU Beds",
    "Number of Pediatric ICU Beds Occupied",
    "Number of Pediatric Inpatient Beds Occupied",
    "Number of Pediatric Inpatient beds",
  ],
  "Bed Capacity (%)": [
    "Percent ICU Beds Occupied",
    "Percent ICU Beds Occupied by COVID-19 Patients",
    "Percent ICU Beds Occupied by Influenza Patients",
    "Percent ICU Beds Occupied by RSV Patients",
    "Percent Inpatient Beds Occupied",
    "Percent Inpatient Beds Occupied by COVID-19 Patients",
    "Percent Inpatient Beds Occupied by Influenza Patients",
    "Percent Inpatient Beds Occupied by RSV Patients",
    "Percent Adult Inpatient Beds Occupied",
    "Percent Pediatric Inpatient Beds Occupied",
    "Percent Adult ICU Beds Occupied",
    "Percent Pediatric ICU Beds Occupied",
  ],
};

// longform:shortform, this may have random column names that don't actually exist
export const nhsnNameToSlugMap = {
  "Week Ending Date": "weekendingdate",
  "Geographic aggregation": "jurisdiction",
  "Number of Inpatient Beds": "numinptbeds",
  "Number of Adult Inpatient Beds": "numinptbedsadult",
  "Number of Pediatric Inpatient beds": "numinptbedsped",
  "Number of Inpatient Beds Occupied": "numinptbedsocc",
  "Number of Adult Inpatient Beds Occupied": "numinptbedsoccadult",
  "Number of Pediatric Inpatient Beds Occupied": "numinptbedsoccped",
  "Number of ICU Beds": "numicubeds",
  "Number of Adult ICU Beds": "numicubedsadult",
  "Number of Pediatric ICU Beds": "numicubedsped",
  "Number of ICU Beds Occupied": "numicubedsocc",
  "Number of Adult ICU Beds Occupied": "numicubedsoccadult",
  "Number of Pediatric ICU Beds Occupied": "numicubedsoccped",
  "Number of Adult Patients Hospitalized with COVID-19":
    "numconfc19hosppatsadult",
  "Number of Pediatric Patients Hospitalized with COVID-19":
    "numconfc19hosppatsped",
  "Total Patients Hospitalized with COVID-19": "totalconfc19hosppats",
  "Number of Adult Patients Hospitalized with Influenza":
    "numconffluhosppatsadult",
  "Number of Pedatric Patients Hospitalized with Influenza":
    "numconffluhosppatsped",
  "Total Patients Hospitalized with Influenza": "totalconffluhosppats",
  "Number of Adult Patients Hospitalized with RSV": "numconfrsvhosppatsadult",
  "Number of Pediatric Patients Hospitalized with RSV": "numconfrsvhosppatsped",
  "Total Patients Hospitalized with RSV": "totalconfrsvhosppats",
  "Number of Adult ICU Patients Hospitalized with COVID-19":
    "numconfc19icupatsadult",
  "Number of Pediatric ICU Patients Hospitalized with COVID-19":
    "numconfc19icupatsped",
  "Total ICU Patients Hospitalized with COVID-19 ": "totalconfc19icupats",
  "Number of Adult ICU Patients Hospitalized with Influenza":
    "numconffluicupatsadult",
  "Number of Pediatric ICU Patients Hospitalized with Influenza":
    "numconffluicupatsped",
  "Total ICU Patients Hospitalized with Influenza": "totalconffluicupats",
  "Number of Adult ICU Patients Hospitalized with RSV":
    "numconfrsvicupatsadult",
  "Number of Pediatric ICU Patients Hospitalized with RSV":
    "numconfrsvicupatsped",
  "Total ICU Patients Hospitalized with RSV": "totalconfrsvicupats",
  "Number of Pediatric COVID-19 Admissions, 0-4 years":
    "numconfc19newadmped0to4",
  "Number of Pediatric COVID-19 Admissions, 5-17 years":
    "numconfc19newadmped5to17",
  "Total Pediatric COVID-19 Admissions": "totalconfc19newadmped",
  "Number of Adult COVID-19 Admissions, 18-49 years":
    "numconfc19newadmadult18to49",
  "Number of Adult COVID-19 Admissions, 50-64 years":
    "numconfc19newadmadult50to64",
  "Number of Adult COVID-19 Admissions, 65-74 years":
    "numconfc19newadmadult65to74",
  "Number of Adult COVID-19 Admissions, 75+ years":
    "numconfc19newadmadult75plus",
  "Total Adult COVID-19 Admissions": "totalconfc19newadmadult",
  "Number of COVID-19 Admissions, unknown age": "numconfc19newadmunk",
  "Total COVID-19 Admissions": "totalconfc19newadm",
  "Number of Pediatric Influenza Admissions, 0-4 years":
    "numconfflunewadmped0to4",
  "Number of Pediatric Influenza Admissions, 5-17 years":
    "numconfflunewadmped5to17",
  "Total Pediatric Influenza Admissions": "totalconfflunewadmped",
  "Number of Adult Influenza Admissions, 18-49 years":
    "numconfflunewadmadult18to49",
  "Number of Adult Influenza Admissions, 50-64 years":
    "numconfflunewadmadult50to64",
  "Number of Adult Influenza Admissions, 65-74 years":
    "numconfflunewadmadult65to74",
  "Number of Adult Influenza Admissions, 75+ years":
    "numconfflunewadmadult75plus",
  "Total Adult Influenza Admissions": "totalconfflunewadmadult",
  "Number of Influenza Admissions, unknown age": "numconfflunewadmunk",
  "Total Influenza Admissions": "totalconfflunewadm",
  "Number of Pediatric RSV Admissions, 0-4 years": "numconfrsvnewadmped0to4",
  "Number of Pediatric RSV Admissions, 5-17 years": "numconfrsvnewadmped5to17",
  "Total Pediatric RSV Admissions": "totalconfrsvnewadmped",
  "Number of Adult RSV Admissions, 18-49 years": "numconfrsvnewadmadult18to49",
  "Number of Adult RSV Admissions, 50-64 years": "numconfrsvnewadmadult50to64",
  "Number of Adult RSV Admissions, 65-74 years": "numconfrsvnewadmadult65to74",
  "Number of Adult RSV Admissions, 75+ years": "numconfrsvnewadmadult75plus",
  "Total Adult RSV Admissions": "totalconfrsvnewadmadult",
  "Number of RSV Admissions, unknown age": "numconfrsvnewadmunk",
  "Total RSV Admissions": "totalconfrsvnewadm",
  "Percent Inpatient Beds Occupied": "pctinptbedsocc",
  "Percent Inpatient Beds Occupied by COVID-19 Patients": "pctconfc19inptbeds",
  "Percent Inpatient Beds Occupied by Influenza Patients": "pctconffluinptbeds",
  "Percent Inpatient Beds Occupied by RSV Patients": "pctconfrsvinptbeds",
  "Percent ICU Beds Occupied": "pcticubedsocc",
  "Percent ICU Beds Occupied by COVID-19 Patients": "pctconfc19icubeds",
  "Percent ICU Beds Occupied by Influenza Patients": "pctconffluicubeds",
  "Percent ICU Beds Occupied by RSV Patients": "pctconfrsvicubeds",
  "Percent Adult COVID-19 Admissions": "pctconfc19newadmadult",
  "Percent Pediatric COVID-19 Admissions": "pctconfc19newadmped",
  "Percent Adult Influenza Admissions": "pctconfflunewadmadult",
  "Percent Pediatric Influenza Admissions": "pctconfflunewadmped",
  "Percent Adult RSV Admissions": "pctconfrsvnewadmadult",
  "Percent Pediatric RSV Admissions": "pctconfrsvnewadmped",
  "Number Hospitals Reporting Number of Inpatient Beds": "numinptbedshosprep",
  "Number Hospitals Reporting Number of Inpatient Beds Occupied":
    "numinptbedsocchosprep",
  "Number Hospitals Reporting Number of ICU Beds ": "numicubedshosprep",
  "Number Hospitals Reporting Number of ICU Beds Occupied":
    "numicubedsocchosprep",
  "Number Hospitals Reporting Total Patients Hospitalized with COVID-19":
    "totalconfc19hosppatshosprep",
  "Number Hospitals Reporting TotalPatients Hospitalized with Influenza":
    "totalconffluhosppatshosprep",
  "Number Hospitals Reporting Total Patients Hospitalized with RSV":
    "totalconfrsvhosppatshosprep",
  "Number Hospitals Reporting ICU Patients Hospitalized with COVID-19 ":
    "totalconfc19icupatshosprep",
  "Number Hospitals Reporting ICU Patients Hospitalized with Influenza":
    "totalconffluicupatshosprep",
  "Number Hospitals Reporting ICU Patients Hospitalized with RSV":
    "totalconfrsvicupatshosprep",
  "Number Hospitals Reporting Pediatric COVID-19 Admissions":
    "totalconfc19newadmpedhosprep",
  "Number Hospitals Reporting Adult COVID-19 Admissions":
    "totalconfc19newadmadulthosprep",
  "Number Hospitals Reporting COVID-19 Admissions": "totalconfc19newadmhosprep",
  "Number Hospitals Reporting Pediatric Influenza Admissions":
    "totalconfflunewadmpedhosprep",
  "Number Hospitals Reporting Adult Influenza Admissions":
    "totalconfflunewadmadulthosprep",
  "Number Hospitals Reporting Influenza Admissions":
    "totalconfflunewadmhosprep",
  "Number Hospitals Reporting Pediatric RSV Admissions":
    "totalconfrsvnewadmpedhosprep",
  "Number Hospitals Reporting Adult RSV Admissions":
    "totalconfrsvnewadmadulthosprep",
  "Number Hospitals Reporting RSV Admissions": "totalconfrsvnewadmhosprep",
  "Number Hospitals Reporting Percent Inpatient Beds Occupied":
    "pctinptbedsocchosprep",
  "Number Hospitals Reporting Percent ICU Beds Occupied":
    "pcticubedsocchosprep",
  "Number Hospitals Reporting Percent Inpatient Beds Occupied by COVID-19 Patients":
    "pctconfc19inptbedshosprep",
  "Number Hospitals Reporting Percent Inpatient Beds Occupied by Influenza Patients":
    "pctconffluinptbedshosprep",
  "Number Hospitals Reporting Percent Inpatient Beds Occupied by RSV Patients":
    "pctconfrsvinptbedshosprep",
  "Number Hospitals Reporting Percent ICU Beds Occupied by COVID-19 Patients":
    "pctconfc19icubedshosprep",
  "Number Hospitals Reporting Percent ICU Beds Occupied by Influenza Patients":
    "pctconffluicubedshosprep",
  "Number Hospitals Reporting Percent ICU Beds Occupied by RSV Patients":
    "pctconfrsvicubedshosprep",
  "Percent Hospitals Reporting Number of Inpatient Beds":
    "numinptbedsperchosprep",
  "Percent Hospitals Reporting Number of Inpatient Beds Occupied":
    "numinptbedsoccperchosprep",
  "Percent Hospitals Reporting Number of ICU Beds ": "numicubedsperchosprep",
  "Percent Hospitals Reporting Number of ICU Beds Occupied":
    "numicubedsoccperchosprep",
  "Percent Hospitals Reporting Total Patients Hospitalized with COVID-19":
    "totalconfc19hosppatsperc",
  "Percent Hospitals Reporting TotalPatients Hospitalized with Influenza":
    "totalconffluhosppatsperc",
  "Percent Hospitals Reporting Total Patients Hospitalized with RSV":
    "totalconfrsvhosppatsperc",
  "Percent Hospitals Reporting ICU Patients Hospitalized with COVID-19 ":
    "totalconfc19icupatsperchosprep",
  "Percent Hospitals Reporting ICU Patients Hospitalized with Influenza":
    "totalconffluicupatsperchosprep",
  "Percent Hospitals Reporting ICU Patients Hospitalized with RSV":
    "totalconfrsvicupatsperchosprep",
  "Percent Hospitals Reporting Pediatric COVID-19 Admissions":
    "totalconfc19newadmpedper",
  "Percent Hospitals Reporting Adult COVID-19 Admissions":
    "totalconfc19newadmadultp",
  "Percent Hospitals Reporting COVID-19 Admissions":
    "totalconfc19newadmperchosprep",
  "Percent Hospitals Reporting Pediatric Influenza Admissions":
    "totalconfflunewadmpedper",
  "Percent Hospitals Reporting Adult Influenza Admissions":
    "totalconfflunewadmadultp",
  "Percent Hospitals Reporting Influenza Admissions":
    "totalconfflunewadmperchosprep",
  "Percent Hospitals Reporting Pediatric RSV Admissions":
    "totalconfrsvnewadmpedper",
  "Percent Hospitals Reporting Adult RSV Admissions":
    "totalconfrsvnewadmadultp",
  "Percent Hospitals Reporting RSV Admissions": "totalconfrsvnewadmperchosprep",
  "Percent Hospitals Reporting Percent Inpatient Beds Occupied":
    "pctinptbedsoccperchosprep",
  "Percent Hospitals Reporting Percent ICU Beds Occupied":
    "pcticubedsoccperchosprep",
  "Percent Hospitals Reporting Percent Inpatient Beds Occupied by COVID-19 Patients":
    "pctconfc19inptbedsperchosprep",
  "Percent Hospitals Reporting Percent Inpatient Beds Occupied by Influenza Patients":
    "pctconffluinptbedsperchosprep",
  "Percent Hospitals Reporting Percent Inpatient Beds Occupied by RSV Patients":
    "pctconfrsvinptbedsperchosprep",
  "Percent Hospitals Reporting Percent ICU Beds Occupied by COVID-19 Patients":
    "pctconfc19icubedsperchosprep",
  "Percent Hospitals Reporting Percent ICU Beds Occupied by Influenza Patients":
    "pctconffluicubedsperchosprep",
  "Percent Hospitals Reporting Percent ICU Beds Occupied by RSV Patients":
    "pctconfrsvicubedsperchosprep",
  "Absolute Change in the Percent Hospitals Reporting Number of Inpatient Beds from Prior Week":
    "numinptbedsperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Number of Inpatient Beds Occupied from Prior Week":
    "numinptbedsoccperchospre",
  "Absolute Change in the Percent Hospitals Reporting Number of ICU Beds from Prior Week":
    "numicubedsperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Number of ICU Beds Occupied from Prior Week":
    "numicubedsoccperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Total Patients Hospitalized with COVID-19 from Prior Week":
    "totalconfc19hosppatsperc_1",
  "Absolute Change in the Percent Hospitals Reporting TotalPatients Hospitalized with Influenza from Prior Week":
    "totalconffluhosppatsperc_1",
  "Absolute Change in the Percent Hospitals Reporting Total Patients Hospitalized with RSV from Prior Week":
    "totalconfrsvhosppatsperc_1",
  "Absolute Change in the Percent Hospitals Reporting ICU Patients Hospitalized with COVID-19 from Prior Week":
    "totalconfc19icupatsperch",
  "Absolute Change in the Percent Hospitals Reporting ICU Patients Hospitalized with Influenza from Prior Week":
    "totalconffluicupatsperch",
  "Absolute Change in the Percent Hospitals Reporting ICU Patients Hospitalized with RSV from Prior Week":
    "totalconfrsvicupatsperch",
  "Absolute Change in the Percent Hospitals Reporting Pediatric COVID-19 Admissions from Prior Week":
    "totalconfc19newadmpedper_1",
  "Absolute Change in the Percent Hospitals Reporting Adult COVID-19 Admissions from Prior Week":
    "totalconfc19newadmadultp_1",
  "Absolute Change in the Percent Hospitals Reporting COVID-19 Admissions from Prior Week":
    "totalconfc19newadmpercho",
  "Absolute Change in the Percent Hospitals Reporting Pediatric Influenza Admissions from Prior Week":
    "totalconfflunewadmpedper_1",
  "Absolute Change in the Percent Hospitals Reporting Adult Influenza Admissions from Prior Week":
    "totalconfflunewadmadultp_1",
  "Absolute Change in the Percent Hospitals Reporting  Influenza Admissions from Prior Week":
    "totalconfflunewadmpercho",
  "Absolute Change in the Percent Hospitals Reporting Pediatric RSV Admissions from Prior Week":
    "totalconfrsvnewadmpedper_1",
  "Absolute Change in the Percent Hospitals Reporting Adult RSV Admissions from Prior Week":
    "totalconfrsvnewadmadultp_1",
  "Absolute Change in the Percent Hospitals Reporting RSV Admissions from Prior Week":
    "totalconfrsvnewadmpercho",
  "Absolute Change in the Percent Hospitals Reporting Percent Inpatient Beds Occupied from Prior Week":
    "pctinptbedsoccperchospre",
  "Absolute Change in the Percent Hospitals Reporting Percent ICU Beds Occupied from Prior Week":
    "pcticubedsoccperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Inpatient Beds Occupied by COVID-19 Patients from Prior Week":
    "pctconfc19inptbedspercho",
  "Absolute Change in the Percent Hospitals Reporting Percent Inpatient Beds Occupied by Influenza Patients from Prior Week":
    "pctconffluinptbedspercho",
  "Absolute Change in the Percent Hospitals Reporting Percent Inpatient Beds Occupied by RSV Patients from Prior Week":
    "pctconfrsvinptbedspercho",
  "Absolute Change in the Percent Hospitals Reporting Percent ICU Beds Occupied by COVID-19 Patients from Prior Week":
    "pctconfc19icubedsperchos",
  "Absolute Change in the Percent Hospitals Reporting Percent ICU Beds Occupied by Influenza Patients from Prior Week":
    "pctconffluicubedsperchos",
  "Absolute Change in the Percent Hospitals Reporting Percent ICU Beds Occupied by RSV Patients from Prior Week":
    "pctconfrsvicubedsperchos",
  "Number of Pediatric COVID-19 Admissions, 0-4 years, per 100,000 population":
    "numconfc19newadmped0to4per100k",
  "Number of Pediatric COVID-19 Admissions, 5-17 years, per 100,000 population":
    "numconfc19newadmped5to17per100k",
  "Total Number of Pediatric COVID-19 Admissions per 100,000 population":
    "totalconfc19newadmpedper100k",
  "Number of Adult COVID-19 Admissions, 18-49 years, per 100,000 population":
    "numconfc19newadmadult18to49per100k",
  "Number of Adult COVID-19 Admissions, 50-64 years, per 100,000 population":
    "numconfc19newadmadult50to64per100k",
  "Number of Adult COVID-19 Admissions, 65-74 years, per 100,000 population":
    "numconfc19newadmadult65to74per100k",
  "Number of Adult COVID-19 Admissions, 75+ years, per 100,000 population":
    "numconfc19newadmadult75plusper100k",
  "Total Number of Adult COVID-19 Admissions per 100,000 population":
    "totalconfc19newadmadultper100k",
  "Total number of COVID-19 Admissions per 100,000 population":
    "totalconfc19newadmper100k",
  "Number of Pediatric Influenza Admissions, 0-4 years, per 100,000 population":
    "numconfflunewadmped0to4per100k",
  "Number of Pediatric Influenza Admissions, 5-17 years, per 100,000 population":
    "numconfflunewadmped5to17per100k",
  "Total Number of Pediatric Influenza Admissions per 100,000 population":
    "totalconfflunewadmpedper100k",
  "Number of Adult Influenza Admissions, 18-49 years, per 100,000 population":
    "numconfflunewadmadult18to49per100k",
  "Number of Adult Influenza Admissions, 50-64 years, per 100,000 population":
    "numconfflunewadmadult50to64per100k",
  "Number of Adult Influenza Admissions, 65-74 years, per 100,000 population":
    "numconfflunewadmadult65to74per100k",
  "Number of Adult Influenza Admissions, 75+ years, per 100,000 population":
    "numconfflunewadmadult75plusper100k",
  "Total Number of Adult Influenza Admissions per 100,000 population":
    "totalconfflunewadmadultper100k",
  "Total number of Influenza Admissions per 100,000 population":
    "totalconfflunewadmper100k",
  "Number of Pediatric RSV Admissions, 0-4 years, per 100,000 population":
    "numconfrsvnewadmped0to4per100k",
  "Number of Pediatric RSV Admissions, 5-17 years, per 100,000 population":
    "numconfrsvnewadmped5to17per100k",
  "Total Number of Pediatric RSV Admissions per 100,000 population":
    "totalconfrsvnewadmpedper100k",
  "Number of Adult RSV Admissions, 18-49 years, per 100,000 population":
    "numconfrsvnewadmadult18to49per100k",
  "Number of Adult RSV Admissions, 50-64 years, per 100,000 population":
    "numconfrsvnewadmadult50to64per100k",
  "Number of Adult RSV Admissions, 65-74 years, per 100,000 population":
    "numconfrsvnewadmadult65to74per100k",
  "Number of Adult RSV Admissions, 75+ years, per 100,000 population":
    "numconfrsvnewadmadult75plusper100k",
  "Total Number of Adult RSV Admissions per 100,000 population":
    "totalconfrsvnewadmadultper100k",
  "Total number of RSV Admissions per 100,000 population":
    "totalconfrsvnewadmper100k",
  "Percent Hospitals Reporting Total New COVID-19 Admissions Above 80%":
    "totalconfc19newadmperchosprepabove80pct",
  "Percent Hospitals Reporting Total New COVID-19 Admissions Above 90%":
    "totalconfc19newadmperchosprepabove90pct",
  "Percent Hospitals Reporting Total New Influenza Admissions Above 80%":
    "totalconfflunewadmperchosprepabove80pct",
  "Percent Hospitals Reporting Total New Influenza Admissions Above 90%":
    "totalconfflunewadmperchosprepabove90pct",
  "Percent Hospitals Reporting Total New RSV Admissions Above 80%":
    "totalconfrsvnewadmperchosprepabove80pct",
  "Percent Hospitals Reporting Total New RSV Admissions Above 90%":
    "totalconfrsvnewadmperchosprepabove90pct",
  "Percent Adult Inpatient Beds Occupied": "pctinptbedsoccadult",
  "Percent Pediatric Inpatient Beds Occupied": "pctinptbedsoccped",
  "Percent Adult ICU Beds Occupied": "pcticubedsoccadult",
  "Percent Pediatric ICU Beds Occupied": "pcticubedsoccped",
  "Percent Adult Inpatient Beds Occupied by COVID-19 Patients":
    "pctconfc19inptbedsadult",
  "Percent Pediatric Inpatient Beds Occupied by COVID-19 Patients":
    "pctconfc19inptbedsped",
  "Percent Adult ICU Beds Occupied by COVID-19 Patients":
    "pctconfc19icubedsadult",
  "Percent Pediatric ICU Beds Occupied by COVID-19 Patients":
    "pctconfc19icubedsped",
  "Percent Adult Inpatient Beds Occupied by Influenza Patients":
    "pctconffluinptbedsadult",
  "Percent Pediatric Inpatient Beds Occupied by Influenza Patients":
    "pctconffluinptbedsped",
  "Percent Adult ICU Beds Occupied by Influenza Patients":
    "pctconffluicubedsadult",
  "Percent Pediatric ICU Beds Occupied by Influenza Patients":
    "pctconffluicubedsped",
  "Percent Adult Inpatient Beds Occupied by RSV Patients":
    "pctconfrsvinptbedsadult",
  "Percent Pediatric Inpatient Beds Occupied by RSV Patients":
    "pctconfrsvinptbedsped",
  "Percent Adult ICU Beds Occupied by RSV Patients": "pctconfrsvicubedsadult",
  "Percent Pediatric ICU Beds Occupied by RSV Patients": "pctconfrsvicubedsped",
  "Percent Hospitalized Patients with COVID-19 in the ICU":
    "pctconfc19hosppatsicu",
  "Percent Hospitalized Adult Patients with COVID-19 in the ICU":
    "pctconfc19hosppatsicuadult",
  "Percent Hospitalized Pediatric Patients with COVID-19 in the ICU":
    "pctconfc19hosppatsicuped",
  "Percent Hospitalized Patients with Influenza in the ICU":
    "pctconffluhosppatsicu",
  "Percent Hospitalized Adult Patients with Influenza in the ICU":
    "pctconffluhosppatsicuadult",
  "Percent Hospitalized Pediatric Patients with Influenza in the ICU":
    "pctconffluhosppatsicuped",
  "Percent Hospitalized Patients with RSV in the ICU": "pctconfrsvhosppatsicu",
  "Percent Hospitalized Adult Patients with RSV in the ICU":
    "pctconfrsvhosppatsicuadult",
  "Percent Hospitalized Pediatric Patients with RSV in the ICU":
    "pctconfrsvhosppatsicuped",
  "Percent Change in the Number of Pediatric COVID-19 Admissions, 0-4 years from Prior Week":
    "numconfc19newadmped0to4pctchg",
  "Percent Change in the Number of Pediatric COVID-19 Admissions, 5-17 years from Prior Week":
    "numconfc19newadmped5to17pctchg",
  "Percent Change in the Total Pediatric COVID-19 Admissions from Prior Week":
    "totalconfc19newadmpedpctchg",
  "Percent Change in the Number of Adult COVID-19 Admissions, 18-49 years from Prior Week":
    "numconfc19newadmadult18to49pctchg",
  "Percent Change in the Number of Adult COVID-19 Admissions, 50-64 years from Prior Week":
    "numconfc19newadmadult50to64pctchg",
  "Percent Change in the Number of Adult COVID-19 Admissions, 65-74 years from Prior Week":
    "numconfc19newadmadult65to74pctchg",
  "Percent Change in the Number of Adult COVID-19 Admissions, 75+ years from Prior Week":
    "numconfc19newadmadult75pluspctchg",
  "Percent Change in the Total Adult COVID-19 Admissions from Prior Week":
    "totalconfc19newadmadultpctchg",
  "Percent Change in the Total COVID-19 Admissions from Prior Week":
    "totalconfc19newadmpctchg",
  "Percent Change in the Number of Pediatric Influenza Admissions, 0-4 years from Prior Week":
    "numconfflunewadmped0to4pctchg",
  "Percent Change in the Number of Pediatric Influenza Admissions, 5-17 years from Prior Week":
    "numconfflunewadmped5to17pctchg",
  "Percent Change in the Total Pediatric Influenza Admissions from Prior Week":
    "totalconfflunewadmpedpctchg",
  "Percent Change in the Number of Adult Influenza Admissions, 18-49 years from Prior Week":
    "numconfflunewadmadult18to49pctchg",
  "Percent Change in the Number of Adult Influenza Admissions, 50-64 years from Prior Week":
    "numconfflunewadmadult50to64pctchg",
  "Percent Change in the Number of Adult Influenza Admissions, 65-74 years from Prior Week":
    "numconfflunewadmadult65to74pctchg",
  "Percent Change in the Number of Adult Influenza Admissions, 75+ years from Prior Week":
    "numconfflunewadmadult75pluspctchg",
  "Percent Change in the Total Adult Influenza Admissions from Prior Week":
    "totalconfflunewadmadultpctchg",
  "Percent Change in the Total Influenza Admissions from Prior Week":
    "totalconfflunewadmpctchg",
  "Percent Change in the Number of Pediatric RSV Admissions, 0-4 years from Prior Week":
    "numconfrsvnewadmped0to4pctchg",
  "Percent Change in the Number of Pediatric RSV Admissions, 5-17 years from Prior Week":
    "numconfrsvnewadmped5to17pctchg",
  "Percent Change in the Total Pediatric RSV Admissions from Prior Week":
    "totalconfrsvnewadmpedpctchg",
  "Percent Change in the Number of Adult RSV Admissions, 18-49 years from Prior Week":
    "numconfrsvnewadmadult18to49pctchg",
  "Percent Change in the Number of Adult RSV Admissions, 50-64 years from Prior Week":
    "numconfrsvnewadmadult50to64pctchg",
  "Percent Change in the Number of Adult RSV Admissions, 65-74 years from Prior Week":
    "numconfrsvnewadmadult65to74pctchg",
  "Percent Change in the Number of Adult RSV Admissions, 75+ years from Prior Week":
    "numconfrsvnewadmadult75pluspctchg",
  "Percent Change in the Total Adult RSV Admissions from Prior Week":
    "totalconfrsvnewadmadultpctchg",
  "Percent Change in the Total RSV Admissions from Prior Week":
    "totalconfrsvnewadmpctchg",
  "Number Hospitals Reporting Percent Adult Inpatient Beds Occupied":
    "pctinptbedsoccadulthosprep",
  "Number Hospitals Reporting Percent Pediatric Inpatient Beds Occupied":
    "pctinptbedsoccpedhosprep",
  "Number Hospitals Reporting Percent Adult ICU Beds Occupied":
    "pcticubedsoccadulthosprep",
  "Number Hospitals Reporting Percent Pediatric ICU Beds Occupied":
    "pcticubedsoccpedhosprep",
  "Number Hospitals Reporting Percent Adult Inpatient Beds Occupied by COVID-19 Patients":
    "pctconfc19inptbedsadulthosprep",
  "Number Hospitals Reporting Percent Pediatric Inpatient Beds Occupied by COVID-19 Patients":
    "pctconfc19inptbedspedhosprep",
  "Number Hospitals Reporting Percent Adult ICU Beds Occupied by COVID-19 Patients":
    "pctconfc19icubedsadulthosprep",
  "Number Hospitals Reporting Percent Pediatric ICU Beds Occupied by COVID-19 Patients":
    "pctconfc19icubedspedhosprep",
  "Number Hospitals Reporting Percent Adult Inpatient Beds Occupied by Influenza Patients":
    "pctconffluinptbedsadulthosprep",
  "Number Hospitals Reporting Percent Pediatric Inpatient Beds Occupied by Influenza Patients":
    "pctconffluinptbedspedhosprep",
  "Number Hospitals Reporting Percent Adult ICU Beds Occupied by Influenza Patients":
    "pctconffluicubedsadulthosprep",
  "Number Hospitals Reporting Percent Pediatric ICU Beds Occupied by Influenza Patients":
    "pctconffluicubedspedhosprep",
  "Number Hospitals Reporting Percent Adult Inpatient Beds Occupied by RSV Patients":
    "pctconfrsvinptbedsadulthosprep",
  "Number Hospitals Reporting Percent Pediatric Inpatient Beds Occupied by RSV Patients":
    "pctconfrsvinptbedspedhosprep",
  "Number Hospitals Reporting Percent Adult ICU Beds Occupied by RSV Patients":
    "pctconfrsvicubedsadulthosprep",
  "Number Hospitals Reporting Percent Pediatric ICU Beds Occupied by RSV Patients":
    "pctconfrsvicubedspedhosprep",
  "Number Hospitals Reporting Percent Hospitalized Patients with COVID-19 in the ICU":
    "pctconfc19hosppatsicuhosprep",
  "Number Hospitals Reporting Percent Hospitalized Adult Patients with COVID-19 in the ICU":
    "pctconfc19hosppatsicuadulthosprep",
  "Number Hospitals Reporting Percent Hospitalized Pediatric Patients with COVID-19 in the ICU":
    "pctconfc19hosppatsicupedhosprep",
  "Number Hospitals Reporting Percent Hospitalized Patients with Influenza in the ICU":
    "pctconffluhosppatsicuhosprep",
  "Number Hospitals Reporting Percent Hospitalized Adult Patients with Influenza in the ICU":
    "pctconffluhosppatsicuadulthosprep",
  "Number Hospitals Reporting Percent Hospitalized Pediatric Patients with Influenza in the ICU":
    "pctconffluhosppatsicupedhosprep",
  "Number Hospitals Reporting Percent Hospitalized Patients with RSV in the ICU":
    "pctconfrsvhosppatsicuhosprep",
  "Number Hospitals Reporting Percent Hospitalized Adult Patients with RSV in the ICU":
    "pctconfrsvhosppatsicuadulthosprep",
  "Number Hospitals Reporting Percent Hospitalized Pediatric Patients with RSV in the ICU":
    "pctconfrsvhosppatsicupedhosprep",
  "Percent Hospitals Reporting Percent Adult Inpatient Beds Occupied":
    "pctinptbedsoccadultperchosprep",
  "Percent Hospitals Reporting Percent Pediatric Inpatient Beds Occupied":
    "pctinptbedsoccpedperchosprep",
  "Percent Hospitals Reporting Percent Adult ICU Beds Occupied":
    "pcticubedsoccadultperchosprep",
  "Percent Hospitals Reporting Percent Pediatric ICU Beds Occupied":
    "pcticubedsoccpedperchosprep",
  "Percent Hospitals Reporting Percent Adult Inpatient Beds Occupied by COVID-19 Patients":
    "pctconfc19inptbedsadultperchosprep",
  "Percent Hospitals Reporting Percent Pediatric Inpatient Beds Occupied by COVID-19 Patients":
    "pctconfc19inptbedspedperchosprep",
  "Percent Hospitals Reporting Percent Adult ICU Beds Occupied by COVID-19 Patients":
    "pctconfc19icubedsadultperchosprep",
  "Percent Hospitals Reporting Percent Pediatric ICU Beds Occupied by COVID-19 Patients":
    "pctconfc19icubedspedperchosprep",
  "Percent Hospitals Reporting Percent Adult Inpatient Beds Occupied by Influenza Patients":
    "pctconffluinptbedsadultperchosprep",
  "Percent Hospitals Reporting Percent Pediatric Inpatient Beds Occupied by Influenza Patients":
    "pctconffluinptbedspedperchosprep",
  "Percent Hospitals Reporting Percent Adult ICU Beds Occupied by Influenza Patients":
    "pctconffluicubedsadultperchosprep",
  "Percent Hospitals Reporting Percent Pediatric ICU Beds Occupied by Influenza Patients":
    "pctconffluicubedspedperchosprep",
  "Percent Hospitals Reporting Percent Adult Inpatient Beds Occupied by RSV Patients":
    "pctconfrsvinptbedsadultperchosprep",
  "Percent Hospitals Reporting Percent Pediatric Inpatient Beds Occupied by RSV Patients":
    "pctconfrsvinptbedspedperchosprep",
  "Percent Hospitals Reporting Percent Adult ICU Beds Occupied by RSV Patients":
    "pctconfrsvicubedsadultperchosprep",
  "Percent Hospitals Reporting Percent Pediatric ICU Beds Occupied by RSV Patients":
    "pctconfrsvicubedspedperchosprep",
  "Percent Hospitals Reporting Percent Hospitalized Patients with COVID-19 in the ICU":
    "pctconfc19hosppatsicuperchosprep",
  "Percent Hospitals Reporting Percent Hospitalized Adult Patients with COVID-19 in the ICU":
    "pctconfc19hosppatsicuadultperchosprep",
  "Percent Hospitals Reporting Percent Hospitalized Pediatric Patients with COVID-19 in the ICU":
    "pctconfc19hosppatsicupedperchosprep",
  "Percent Hospitals Reporting Percent Hospitalized Patients with Influenza in the ICU":
    "pctconffluhosppatsicuperchosprep",
  "Percent Hospitals Reporting Percent Hospitalized Adult Patients with Influenza in the ICU":
    "pctconffluhosppatsicuadultperchosprep",
  "Percent Hospitals Reporting Percent Hospitalized Pediatric Patients with Influenza in the ICU":
    "pctconffluhosppatsicupedperchosprep",
  "Percent Hospitals Reporting Percent Hospitalized Patients with RSV in the ICU":
    "pctconfrsvhosppatsicuperchosprep",
  "Percent Hospitals Reporting Percent Hospitalized Adult Patients with RSV in the ICU":
    "pctconfrsvhosppatsicuadultperchosprep",
  "Percent Hospitals Reporting Percent Hospitalized Pediatric Patients with RSV in the ICU":
    "pctconfrsvhosppatsicupedperchosprep",
  "Absolute Change in the Percent Hospitals Reporting Percent Adult Inpatient Beds Occupied from Prior Week":
    "pctinptbedsoccadultperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Pediatric Inpatient Beds Occupied from Prior Week":
    "pctinptbedsoccpedperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Adult ICU Beds Occupied from Prior Week":
    "pcticubedsoccadultperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Pediatric ICU Beds Occupied from Prior Week":
    "pcticubedsoccpedperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Adult Inpatient Beds Occupied by COVID-19 Patients from Prior Week":
    "pctconfc19inptbedsadultperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Pediatric Inpatient Beds Occupied by COVID-19 Patients from Prior Week":
    "pctconfc19inptbedspedperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Adult ICU Beds Occupied by COVID-19 Patients from Prior Week":
    "pctconfc19icubedsadultperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Pediatric ICU Beds Occupied by COVID-19 Patients from Prior Week":
    "pctconfc19icubedspedperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Adult Inpatient Beds Occupied by Influenza Patients from Prior Week":
    "pctconffluinptbedsadultperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Pediatric Inpatient Beds Occupied by Influenza Patients from Prior Week":
    "pctconffluinptbedspedperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Adult ICU Beds Occupied by Influenza Patients from Prior Week":
    "pctconffluicubedsadultperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Pediatric ICU Beds Occupied by Influenza Patients from Prior Week":
    "pctconffluicubedspedperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Adult Inpatient Beds Occupied by RSV Patients from Prior Week":
    "pctconfrsvinptbedsadultperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Pediatric Inpatient Beds Occupied by RSV Patients from Prior Week":
    "pctconfrsvinptbedspedperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Adult ICU Beds Occupied by RSV Patients from Prior Week":
    "pctconfrsvicubedsadultperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Pediatric ICU Beds Occupied by RSV Patients from Prior Week":
    "pctconfrsvicubedspedperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Hospitalized Patients with COVID-19 in the ICU from Prior Week":
    "pctconfc19hosppatsicuperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Hospitalized Adult Patients with COVID-19 in the ICU from Prior Week":
    "pctconfc19hosppatsicuadultperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Hospitalized Pediatric Patients with COVID-19 in the ICU from Prior Week":
    "pctconfc19hosppatsicupedperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Hospitalized Patients with Influenza in the ICU from Prior Week":
    "pctconffluhosppatsicuperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Hospitalized Adult Patients with Influenza in the ICU from Prior Week":
    "pctconffluhosppatsicuadultperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Hospitalized Pediatric Patients with Influenza in the ICU from Prior Week":
    "pctconffluhosppatsicupedperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Hospitalized Patients with RSV in the ICU from Prior Week":
    "pctconfrsvhosppatsicuperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Hospitalized Adult Patients with RSV in the ICU from Prior Week":
    "pctconfrsvhosppatsicuadultperchosprepabschg",
  "Absolute Change in the Percent Hospitals Reporting Percent Hospitalized Pediatric Patients with RSV in the ICU from Prior Week":
    "pctconfrsvhosppatsicupedperchosprepabschg",
};

// shortform:longform
export const nhsnSlugToNameMap = Object.fromEntries(
  Object.entries(nhsnNameToSlugMap).map(([name, slug]) => [slug, name]),
);

// change names for better appearance in column selector
export const nhsnNameToPrettyNameMap = {
  "Week Ending Date": "weekendingdate",
  "Geographic aggregation": "jurisdiction",

  "Number of Inpatient Beds": "Inpatient Beds",
  "Number of Adult Inpatient Beds": "Adult Inpatient Beds",
  "Number of Pediatric Inpatient beds": "Pediatric Inpatient Beds",
  "Number of Inpatient Beds Occupied": "Inpatient Beds Occupied",
  "Number of Adult Inpatient Beds Occupied": "Adult Inpatient Beds Occupied",
  "Number of Pediatric Inpatient Beds Occupied":
    "Pediatric Inpatient Beds Occupied",
  "Number of ICU Beds": "ICU Beds",
  "Number of Adult ICU Beds": "Adult ICU Beds",
  "Number of Pediatric ICU Beds": "Pediatric ICU Beds",
  "Number of ICU Beds Occupied": "ICU Beds Occupied",
  "Number of Adult ICU Beds Occupied": "Adult ICU Beds Occupied",
  "Number of Pediatric ICU Beds Occupied": "Pediatric ICU Beds Occupied",
  "Number of Adult Patients Hospitalized with COVID-19": "Adult Hospitalized",
  "Number of Pediatric Patients Hospitalized with COVID-19":
    "Pediatric Hospitalized",
  "Total Patients Hospitalized with COVID-19": "Total Hospitalized",
  "Number of Adult Patients Hospitalized with Influenza": "Adult Hospitalized",
  "Number of Pedatric Patients Hospitalized with Influenza":
    "Pediatric Hospitalized",
  "Total Patients Hospitalized with Influenza": "Total Hospitalized",
  "Number of Adult Patients Hospitalized with RSV": "Adult Hospitalized",
  "Number of Pediatric Patients Hospitalized with RSV":
    "Pediatric Hospitalized",
  "Total Patients Hospitalized with RSV": "Total Hospitalized",
  "Number of Adult ICU Patients Hospitalized with COVID-19":
    "Adult ICU Patients",
  "Number of Pediatric ICU Patients Hospitalized with COVID-19":
    "Pediatric ICU Patients",
  "Total ICU Patients Hospitalized with COVID-19 ": "Total ICU Patients",
  "Number of Adult ICU Patients Hospitalized with Influenza":
    "Adult ICU Patients",
  "Number of Pediatric ICU Patients Hospitalized with Influenza":
    "Pediatric ICU Patients",
  "Total ICU Patients Hospitalized with Influenza": "Total ICU Patients",
  "Number of Adult ICU Patients Hospitalized with RSV": "Adult ICU Patients",
  "Number of Pediatric ICU Patients Hospitalized with RSV":
    "Pediatric ICU Patients",
  "Total ICU Patients Hospitalized with RSV": "Total ICU Patients",
  "Number of Pediatric COVID-19 Admissions, 0-4 years":
    "Pediatric Admissions (0-4 years)",
  "Number of Pediatric COVID-19 Admissions, 5-17 years":
    "Pediatric Admissions (5-17 years)",
  "Total Pediatric COVID-19 Admissions": "Pediatric Admissions",
  "Number of Adult COVID-19 Admissions, 18-49 years":
    "Adult Admissions (18-49 years)",
  "Number of Adult COVID-19 Admissions, 50-64 years":
    "Adult Admissions (50-64 years)",
  "Number of Adult COVID-19 Admissions, 65-74 years":
    "Adult Admissions (65-74 years)",
  "Number of Adult COVID-19 Admissions, 75+ years":
    "Adult Admissions (75+ years)",
  "Total Adult COVID-19 Admissions": "Adult Admissions",
  "Number of COVID-19 Admissions, unknown age": "Admissions (unknown age)",
  "Total COVID-19 Admissions": "Total Admissions",
  "Number of Pediatric Influenza Admissions, 0-4 years":
    "Pediatric Admissions (0-4 years)",
  "Number of Pediatric Influenza Admissions, 5-17 years":
    "Pediatric Admissions (5-17 years)",
  "Total Pediatric Influenza Admissions": "Pediatric Admissions",
  "Number of Adult Influenza Admissions, 18-49 years":
    "Adult Admissions (18-49 years)",
  "Number of Adult Influenza Admissions, 50-64 years":
    "Adult Admissions (50-64 years)",
  "Number of Adult Influenza Admissions, 65-74 years":
    "Adult Admissions (65-74 years)",
  "Number of Adult Influenza Admissions, 75+ years":
    "Adult Admissions (75+ years)",
  "Total Adult Influenza Admissions": "Adult Admissions",
  "Number of Influenza Admissions, unknown age": "Admissions (unknown age)",
  "Total Influenza Admissions": "Total Admissions",
  "Number of Pediatric RSV Admissions, 0-4 years":
    "Pediatric Admissions (0-4 years)",
  "Number of Pediatric RSV Admissions, 5-17 years":
    "Pediatric Admissions (5-17 years)",
  "Total Pediatric RSV Admissions": "Pediatric Admissions",
  "Number of Adult RSV Admissions, 18-49 years":
    "Adult Admissions (18-49 years)",
  "Number of Adult RSV Admissions, 50-64 years":
    "Adult Admissions (50-64 years)",
  "Number of Adult RSV Admissions, 65-74 years":
    "Adult Admissions (65-74 years)",
  "Number of Adult RSV Admissions, 75+ years": "Adult Admissions (75+ years)",
  "Total Adult RSV Admissions": "Adult Admissions",
  "Number of RSV Admissions, unknown age": "Admissions (unknown age)",
  "Total RSV Admissions": "Total Admissions",
  "Percent Adult ICU Beds Occupied": "Percent of Adult ICU Beds Occupied",
  "Percent Pediatric ICU Beds Occupied":
    "Percent of Pediatric ICU Beds Occupied",
  "Percent Adult Inpatient Beds Occupied":
    "Percent of Adult Inpatient Beds Occupied",
  "Percent Pediatric Inpatient Beds Occupied":
    "Percent of Pediatric Inpatient Beds Occupied",
  "Percent Inpatient Beds Occupied": "Percent of Inpatient Beds Occupied",
  "Percent Inpatient Beds Occupied by COVID-19 Patients":
    "Percent of Inpatient Beds Occupied by COVID-19 Patients",
  "Percent Inpatient Beds Occupied by Influenza Patients":
    "Percent of Inpatient Beds Occupied by Influenza Patients",
  "Percent Inpatient Beds Occupied by RSV Patients":
    "Percent of Inpatient Beds Occupied by RSV Patients",
  "Percent ICU Beds Occupied": "Percent of ICU Beds Occupied",
  "Percent ICU Beds Occupied by COVID-19 Patients":
    "Percent of ICU Beds Occupied by COVID-19 Patients",
  "Percent ICU Beds Occupied by Influenza Patients":
    "Percent of ICU Beds Occupied by Influenza Patients",
  "Percent ICU Beds Occupied by RSV Patients":
    "Percent of ICU Beds Occupied by RSV Patients",
  "Percent Adult COVID-19 Admissions":
    "Percent of COVID-19 Admissions that are Adults",
  "Percent Pediatric COVID-19 Admissions":
    "Percent of COVID-19 Admissions that are Pediatric",
  "Percent Adult Influenza Admissions":
    "Percent of Influenza Admissions that are Adults",
  "Percent Pediatric Influenza Admissions":
    "Percent of Influenza Admissions that are Pediatric",
  "Percent Adult RSV Admissions": "Percent of RSV Admissions that are Adults",
  "Percent Pediatric RSV Admissions":
    "Percent of RSV Admissions that are Pediatric",
  "Number of Pediatric COVID-19 Admissions, 0-4 years, per 100,000 population":
    "Pediatric Admissions (0-4 years)",
  "Number of Pediatric COVID-19 Admissions, 5-17 years, per 100,000 population":
    "Pediatric Admissions (5-17 years)",
  "Total Number of Pediatric COVID-19 Admissions per 100,000 population":
    "Pediatric Admissions",
  "Number of Adult COVID-19 Admissions, 18-49 years, per 100,000 population":
    "Adult Admissions (18-49 years)",
  "Number of Adult COVID-19 Admissions, 50-64 years, per 100,000 population":
    "Adult Admissions (50-64 years)",
  "Number of Adult COVID-19 Admissions, 65-74 years, per 100,000 population":
    "Adult Admissions (65-74 years)",
  "Number of Adult COVID-19 Admissions, 75+ years, per 100,000 population":
    "Adult Admissions (75+ years)",
  "Total Number of Adult COVID-19 Admissions per 100,000 population":
    "Adult Admissions",
  "Total number of COVID-19 Admissions per 100,000 population":
    "Total Admissions",
  "Number of Pediatric Influenza Admissions, 0-4 years, per 100,000 population":
    "Pediatric Admissions (0-4 years)",
  "Number of Pediatric Influenza Admissions, 5-17 years, per 100,000 population":
    "Pediatric Admissions (5-17 years)",
  "Total Number of Pediatric Influenza Admissions per 100,000 population":
    "Pediatric Admissions",
  "Number of Adult Influenza Admissions, 18-49 years, per 100,000 population":
    "Adult Admissions (18-49 years)",
  "Number of Adult Influenza Admissions, 50-64 years, per 100,000 population":
    "Adult Admissions (50-64 years)",
  "Number of Adult Influenza Admissions, 65-74 years, per 100,000 population":
    "Adult Admissions (65-74 years)",
  "Number of Adult Influenza Admissions, 75+ years, per 100,000 population":
    "Adult Admissions (75+ years)",
  "Total Number of Adult Influenza Admissions per 100,000 population":
    "Adult Admissions",
  "Total number of Influenza Admissions per 100,000 population":
    "Total Admissions",
  "Number of Pediatric RSV Admissions, 0-4 years, per 100,000 population":
    "Pediatric Admissions (0-4 years)",
  "Number of Pediatric RSV Admissions, 5-17 years, per 100,000 population":
    "Pediatric Admissions (5-17 years)",
  "Total Number of Pediatric RSV Admissions per 100,000 population":
    "Pediatric Admissions",
  "Number of Adult RSV Admissions, 18-49 years, per 100,000 population":
    "Adult Admissions (18-49 years)",
  "Number of Adult RSV Admissions, 50-64 years, per 100,000 population":
    "Adult Admissions (50-64 years)",
  "Number of Adult RSV Admissions, 65-74 years, per 100,000 population":
    "Adult Admissions (65-74 years)",
  "Number of Adult RSV Admissions, 75+ years, per 100,000 population":
    "Adult Admissions (75+ years)",
  "Total Number of Adult RSV Admissions per 100,000 population":
    "Adult Admissions",
  "Total number of RSV Admissions per 100,000 population": "Total Admissions",
};
