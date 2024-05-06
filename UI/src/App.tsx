import CorridorSelect from "./components/CorridorSelect";
import PatientViewSelect from "./components/PatientViewSelect";
import { Flex, Stack, Box } from "@chakra-ui/react";
import VisualizationBoxes from "./components/VisualizationBoxes";
import XRayDisplays from "./components/XRayDisplays";

function App() {
    /*
        Primary page. Renders XRayDisplays and control panel.
    */
    return (
        <div>
            <Flex direction="row" align="stretch" w="100%" p="4">
                <Box flex="3" maxHeight='95vh' overflow= 'auto' pl="10rem"> 
                    <XRayDisplays />
                </Box>

                <Flex justifyContent="flex-end" flex="2" >
                    <Box
                        borderRadius="lg"
                        display="inline-flex"
                        flexDirection="column"
                        p="10px"
                        m="2"
                        borderWidth="3px"
                        bg="whiteAlpha.100"
                        maxHeight='95vh'
                        overflow= 'auto'
                    >
                        <Stack spacing="4">
                            <CorridorSelect />
                            <PatientViewSelect />
                            <VisualizationBoxes />
                        </Stack>
                    </Box>
                </Flex>
            </Flex>
        </div>
    );
}
export default App;
