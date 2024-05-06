/*
    Component to display a bone overlay on top of an X-ray image from a set of 
    perimeter-defining points.
*/

interface Props {
    points: [number, number][],
    visible: boolean;
    color: string;
}

function BoneOverlay( {points, visible, color}: Props) {
    if (!visible) {
        return null;
    }
    const pts_string = points.map(point => point.join(',')).join(' ');

    return (
        <polygon
            fill={color}
            stroke={color}
            strokeWidth="3"
            points={pts_string}
            style={{ fillOpacity: 0.04, strokeOpacity: 0.3 }}
        />
    );
}

export default BoneOverlay;