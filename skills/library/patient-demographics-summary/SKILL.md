---
name: patient-demographics-summary
description: |
  Retrieves and formats a complete patient demographic summary from FHIR Patient, RelatedPerson, and Coverage resources.
  Use when user asks to "pull demographics", "get patient info", "show patient details", "who is this patient",
  "patient summary", "emergency contacts", "insurance info", or needs a quick overview of non-clinical patient data.
  Do NOT use for clinical summaries, lab results, medication lists, or diagnosis reviews.
---

# Patient Demographics Summary

## Overview

Pull structured demographic data from FHIR Patient, RelatedPerson, and Coverage resources. Format into a consolidated summary including identifiers, contact information, emergency contacts, insurance, preferred language, communication preferences, and advance directive status. Flag missing critical data elements that may affect care delivery.

## FHIR Resources Used

| Resource | Purpose | Key Fields |
|----------|---------|------------|
| Patient | Core demographics | name, birthDate, gender, address, telecom, communication, identifier, extension |
| RelatedPerson | Emergency contacts, guarantors | name, telecom, relationship, period |
| Coverage | Insurance information | status, type, subscriberId, payor, period, class |

## Tools Available On This Server

This deployment exposes **patient-centric** tools. Call exactly these names.

| Purpose | Tool |
|---------|------|
| Full patient resource | `fhir__get_patient_by_id` |
| Core demographics only | `fhir__get_patient_core_by_id` |
| Names | `fhir__list_patient_names` |
| Identifiers (MRN, SSN) | `fhir__list_patient_identifiers` |
| Phone / email / fax | `fhir__list_patient_telecom` |
| Addresses | `fhir__list_patient_addresses` |
| Preferred languages | `fhir__list_patient_communications` |
| Contacts (next of kin, emergency) | `fhir__list_patient_contacts` |
| Nominated primary care provider | `fhir__list_patient_general_practitioners` |
| Photos | `fhir__list_patient_photos` |
| Links to related patients/persons | `fhir__list_patient_links` |

### Not available on this server

There is **no tool** for Coverage, Consent, DocumentReference, or
AllergyIntolerance. Do not attempt them and do not infer their contents. Report
them under `MISSING DATA FLAGS` as *not available on this server* — do not
report them as "none on file", which would be a different (unsupported) claim.

### Two traps

- **`fhir__list_patient_links` is not the contacts tool.** It returns
  `Patient.link` entries (merged-record / replaced-by relationships between
  patient records), not next-of-kin or emergency contacts. Use
  `fhir__list_patient_contacts` for people.
- **`fhir__list_patient_contacts` returns embedded `Patient.contact`
  entries**, not separate RelatedPerson resources. The relationship, name, and
  telecom of each entry are still authoritative — read them from the entry
  itself.

## Instructions

### Step 1: Retrieve the patient

```
Tool: fhir__get_patient_by_id
id: "[patient-id]"
```

Use `fhir__get_patient_core_by_id` instead when only basic demographics are
needed. Either way, verify the returned resource is the requested patient before
continuing; if it is not, stop and report the mismatch.

Extract the following fields:
- **Identifiers**: MRN (`identifier` where `type.coding.code` = "MR"), SSN (last 4 only), driver's license -- or `fhir__list_patient_identifiers`
- **Name**: `name` array -- use `name.use` = "official" as primary; note maiden names (`use` = "maiden") -- or `fhir__list_patient_names`
- **Birth Date**: `birthDate` (calculate age)
- **Gender**: `gender` (administrative gender)
- **Race/Ethnicity**: US Core extensions at `extension` where URL = `http://hl7.org/fhir/us/core/StructureDefinition/us-core-race` and `http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity`
- **Address**: `address` array -- identify `use` = "home" vs "work" vs "temp" -- or `fhir__list_patient_addresses`
- **Telecom**: `telecom` array -- phone (`system` = "phone"), email (`system` = "email"), fax -- or `fhir__list_patient_telecom`
- **Preferred Language**: `communication` array where `preferred` = true -- or `fhir__list_patient_communications`
- **Marital Status**: `maritalStatus`
- **Deceased**: `deceasedBoolean` or `deceasedDateTime`
- **Managing Organization**: `managingOrganization`

If the full resource comes back sparse, call the per-field `fhir__list_patient_*`
tools before concluding a field is missing.

### Step 2: Retrieve Emergency Contacts and Related Persons

```
Tool: fhir__list_patient_contacts
id: "[patient-id]"
```

For each contact:
- Extract name, relationship (`relationship.coding.code` and text), phone, email
- Check `period` where present to confirm the relationship is current (no `end` date or `end` > today)
- Identify relationship roles from the coded relationship only: emergency contact, next of kin, guarantor, power of attorney

Do not call `fhir__list_patient_links` for this step — see the traps above.

### Step 3: Insurance Coverage -- NOT AVAILABLE

There is no Coverage tool on this server. Do **not** attempt a coverage lookup
and do **not** state or imply whether the patient is insured.

Add to `MISSING DATA FLAGS`:
- `No active insurance coverage — coverage is not available on this server`

### Step 4: Advance Directives -- NOT AVAILABLE

There is no Consent or DocumentReference tool on this server. Do not attempt the
lookup and do not assume a directive exists or does not exist.

Add to `MISSING DATA FLAGS`:
- `No advance directive on file — advance directives are not available on this server`

### Step 5: Nominated Primary Care Provider

```
Tool: fhir__list_patient_general_practitioners
id: "[patient-id]"
```

Report each nominated general practitioner by name and reference. If the list is
empty, flag it as a missing data item.

### Step 6: Flag Missing Critical Data

Scan the collected data and flag any of the following as missing:
- No emergency contact (zero contacts, or none with an emergency-contact relationship)
- No preferred language specified
- Coverage not retrievable on this server (always flag — see Step 3)
- Advance directive not retrievable on this server (always flag — see Step 4)
- Allergy documentation status unknown (no AllergyIntolerance tool on this server)
- No nominated primary care provider
- Missing phone number or email
- Missing address

### Step 7: Format Output

Present the summary in this structure:

```
PATIENT DEMOGRAPHICS
====================
Name: [official name]
MRN: [identifier]
DOB: [birthDate] (Age: [calculated])
Gender: [gender]
Race/Ethnicity: [from extensions]
Preferred Language: [language]
Marital Status: [status]
Nominated GP: [name, or "None recorded"]

CONTACT INFORMATION
===================
Home Phone: [telecom]
Mobile Phone: [telecom]
Email: [telecom]
Home Address: [address]

EMERGENCY CONTACTS
==================
1. [name] - [relationship] - [phone]
2. [name] - [relationship] - [phone]

INSURANCE COVERAGE
==================
Not available on this server

ADVANCE DIRECTIVES
==================
Not available on this server

MISSING DATA FLAGS
==================
- [list of flagged items]
```

## Examples

### Example 1: Complete Demographics Pull

**User says:** "Pull up demographics for patient 12345"

**Actions:**
1. `fhir__get_patient_by_id` id=12345 -- returns name, DOB, address, telecom, US Core race/ethnicity extensions
2. `fhir__list_patient_contacts` id=12345 -- returns 2 entries: spouse (emergency contact), adult child (next of kin)
3. `fhir__list_patient_general_practitioners` id=12345 -- returns 1 nominated GP
4. Coverage and advance directives are not queried (no tool on this server)

**Result:**
```
PATIENT DEMOGRAPHICS
====================
Name: Maria Garcia
MRN: MRN-12345
DOB: 1958-03-15 (Age: 67)
Gender: Female
Race/Ethnicity: Hispanic or Latino / White
Preferred Language: Spanish
Marital Status: Married
Nominated GP: Dr. A. Patel

CONTACT INFORMATION
===================
Home Phone: (555) 234-5678
Mobile Phone: (555) 876-5432
Email: mgarcia@email.com
Home Address: 456 Oak Ave, Austin, TX 78701

EMERGENCY CONTACTS
==================
1. Roberto Garcia - Spouse - (555) 234-5679
2. Ana Garcia-Lopez - Daughter - (555) 345-6789

INSURANCE COVERAGE
==================
Not available on this server

ADVANCE DIRECTIVES
==================
Not available on this server

MISSING DATA FLAGS
==================
- No active insurance coverage - coverage is not available on this server
- No advance directive on file - advance directives are not available on this server
```

### Example 2: Patient With Missing Critical Data

**User says:** "Who is patient abc-789? Get me their info."

**Actions:**
1. `fhir__get_patient_by_id` id=abc-789 -- returns name and DOB only, no telecom, no extensions
2. `fhir__list_patient_contacts` id=abc-789 -- returns 0 entries
3. `fhir__list_patient_general_practitioners` id=abc-789 -- returns 0 entries
4. Coverage and advance directives are not queried (no tool on this server)

**Result:**
```
PATIENT DEMOGRAPHICS
====================
Name: James Wilson
MRN: Not recorded
DOB: 1990-11-22 (Age: 35)
Gender: Male
Race/Ethnicity: Not documented
Preferred Language: Not specified
Marital Status: Unknown
Nominated GP: None recorded

CONTACT INFORMATION
===================
No contact information on file

EMERGENCY CONTACTS
==================
None documented

INSURANCE COVERAGE
==================
Not available on this server

ADVANCE DIRECTIVES
==================
Not available on this server

MISSING DATA FLAGS [CRITICAL]
==============================
- No phone number or email
- No home address
- No emergency contact
- No nominated primary care provider
- No preferred language
- Race/ethnicity not documented
- No active insurance coverage - coverage is not available on this server
- No advance directive on file - advance directives are not available on this server
- Allergy documentation status unknown
```

## Troubleshooting

### No Patient Resource Found
- Verify the patient ID is correct.
- Check if the ID format matches the FHIR server's expectations (some use UUIDs, others use numeric IDs).
- The tools on this server are all keyed by patient id; there is no name-search tool, so a wrong or unknown id cannot be resolved by searching. Report it instead.

### Contact List Is Empty But the Record Mentions a Contact
- Confirm you called `fhir__list_patient_contacts`, not `fhir__list_patient_links`.
- Embedded `Patient.contact` entries may carry relationship codes the server
  does not map; report the raw relationship if you see one.

### Patient Resource Is Sparse
- Call the per-field tools (`fhir__list_patient_names`, `..._identifiers`,
  `..._telecom`, `..._addresses`, `..._communications`) before concluding a
  field is missing. A core read returns fewer fields than a full read.
