/*
    Component to display selection options for patient view (supine or prone)
    using a set of radio buttons.
*/

import { Radio, RadioGroup, Stack, Heading } from "@chakra-ui/react";
import { useData } from '../utility/useContext';

type PatientViewType = 'supine' | 'prone';

function PatientViewSelect() {
    const { patientView, setPatientView } = useData();
    const handleChange = (updatedVal: PatientViewType) => {
        setPatientView(updatedVal);
    };
    return (
        <div>
            <Heading size="lg" mb="2">
                Patient View
            </Heading>
            <RadioGroup onChange={handleChange} value={patientView} size="lg">
                <Stack>
                    <Radio value="supine">Supine</Radio>
                    <Radio value="prone">Prone</Radio>
                </Stack>
            </RadioGroup>
        </div>
    );
}
export default PatientViewSelect;
