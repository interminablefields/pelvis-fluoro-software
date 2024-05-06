/*
    Component to display selection options for corridor using a set of radio buttons.
*/

import { Radio, RadioGroup, Stack, Heading } from "@chakra-ui/react";
import { useData } from '../utility/useContext';

function CorridorSelect() {
    const { selectedCorridor, setSelectedCorridor } = useData();
    const handleChange = (updatedVal: string) => {
        setSelectedCorridor(updatedVal);
    };

    return (
        <div>
            <Heading size="lg" mb="2">
                Corridor
            </Heading>
            <RadioGroup onChange={handleChange} value={selectedCorridor} size="lg">
                <Stack>
                    <Radio value="ramus_right">Right ramus</Radio>
                    <Radio value="ramus_left">Left ramus</Radio>
                    <Radio value="teardrop_right">Right teardrop</Radio>
                    <Radio value="teardrop_left">Left teardrop</Radio>
                    <Radio value="s1">S1</Radio>
                    <Radio value="s2">S2</Radio>
                </Stack>
            </RadioGroup>
        </div>
    );
}
export default CorridorSelect;
