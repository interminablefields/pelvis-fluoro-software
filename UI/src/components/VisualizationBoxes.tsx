/*
    Component to display selection options for visualizations using a set of checkboxes.
    Includes checkboxes to see bone outlines, error bars, the actual X-ray image, the wire
    selection mechanism, and selected corridor.
*/
import { Checkbox, CheckboxGroup, Stack, Heading } from "@chakra-ui/react";
import { useData } from '../utility/useContext';

function VisualizationBoxes() {
    const { showAnatomy, setShowAnatomy, showWireSelect, setShowWireSelect, 
            showCorridor, setShowCorridor, showImage, setShowImage,
            showErrBars, setShowErrBars } = useData();

    const handleAnatomyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setShowAnatomy(e.target.checked);
    };
    const handleErrBarsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setShowErrBars(e.target.checked);
    };
    const handleShowImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setShowImage(e.target.checked);
    };
    const handleWireSelectChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setShowWireSelect(e.target.checked);
    };
    const handleShowCorridor = (e: React.ChangeEvent<HTMLInputElement>) => {
        setShowCorridor(e.target.checked);
    };

    return (
        <div>
            <Heading size="lg" mb="2">
                Visualizations
            </Heading>
            <CheckboxGroup size="lg">
                <Stack>
                    <Checkbox value="outlines" isChecked={showAnatomy} onChange={handleAnatomyChange}>Anatomy Outlines</Checkbox>
                    <Checkbox value="wireselect" isChecked={showWireSelect} onChange={handleWireSelectChange}>Wire Selection</Checkbox>
                    <Checkbox value="imshow" isChecked={showImage} onChange={handleShowImageChange}>Image</Checkbox>
                    <Checkbox value="corridor" isChecked={showCorridor} onChange={handleShowCorridor}>Selected Corridor</Checkbox>
                    <Checkbox value="errbars" isChecked={showErrBars} onChange={handleErrBarsChange}>Error Bars</Checkbox>
                </Stack>
            </CheckboxGroup>
        </div>
    );
}
export default VisualizationBoxes;
