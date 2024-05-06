/*
    Component to display a selectable wire component on top of an X-ray image from a pair of 
    line segment endpoints.
*/
import { useData } from '../utility/useContext';
import { useState } from 'react';
interface Props {
    x1: number;
    y1: number; 
    x2: number;
    y2: number;
    scaleX: number,
    scaleY: number,
    wireKey: string;
    visible: boolean;
    getYawAngle: (coord1: [number, number], coord2: [number, number]) => number;
    setWireCoords: (coords: [[number, number], [number, number]]) => void;
    setWireAngle: (angle: number) => void;
    setWireMidpoint: (midpoint: [number, number]) => void;
}

function WireSegment( {x1, y1, x2, y2, scaleX, scaleY, wireKey, visible, 
                        getYawAngle, setWireCoords, setWireAngle, setWireMidpoint}: Props) {

    const { selectedWire, setSelectedWire } = useData();
    const [hoveredWire, setHoveredWire] = useState<string | null>(null);

    const onClick = () => {
        setSelectedWire(wireKey);
        isSelected = selectedWire === wireKey;
        isHovered = hoveredWire === wireKey;
        console.log("wire selected:", selectedWire);
        const angle = getYawAngle([x1, y1], [x2, y2]);
        setWireCoords([[x1 / scaleX, y1 / scaleY], [x2 / scaleX, y2 / scaleY]]);
        setWireAngle(angle);
        setWireMidpoint([((x1 + x2) / 2)/scaleX, ((y1 + y2) / 2)/scaleY]);
    };

    let isSelected = selectedWire === wireKey;
    let isHovered = hoveredWire === wireKey;
    if (!visible) {
        return null;
    }
    return (
        <line
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke={isSelected ? "#1E90FF" : "#B0C4DE"}
            strokeWidth="5"
            onClick = {onClick}
            onMouseEnter={() => setHoveredWire(wireKey)}
            onMouseLeave={() => setHoveredWire(null)}
            style={{ pointerEvents: 'visibleStroke', cursor: 'pointer', strokeOpacity: isSelected ? 0.8 : isHovered ? 0.7 : 0.5, zIndex: isSelected ? 100 : 1 }}
        />
    );
}

export default WireSegment;